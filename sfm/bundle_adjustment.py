import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


# ─────────────────────────────────────────
#  Rodrigues helpers  (axis-angle ↔ matrix)
# ─────────────────────────────────────────

def mat_to_rvec(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()                     # (3,)

def rvec_to_mat(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    return R                                # (3,3)


# ─────────────────────────────────────────
#  Parameter pack / unpack
# ─────────────────────────────────────────

def pack_params(cameras: dict, tracks: list) -> np.ndarray:
    """
    Flatten all optimisable values into one 1-D vector.

    Camera block  (6 values each):  rvec(3)  tvec(3)
    Point block   (3 values each):  X  Y  Z

    Intrinsics (K) are kept fixed — calibrate separately.
    """
    cam_params = []
    for cam in cameras.values():
        cam_params.append(mat_to_rvec(cam.R))   # 3 values
        cam_params.append(cam.t.ravel())         # 3 values

    pt_params = [t.point3d for t in tracks]     # 3 values each

    return np.concatenate(
        [np.concatenate(cam_params)] +
        [np.array(p) for p in pt_params]
    )


def unpack_params(
    params:   np.ndarray,
    cameras:  dict,
    tracks:   list,
) -> tuple[dict, list]:
    """
    Reverse of pack_params.
    Returns updated (cameras, tracks) without mutating originals.
    """
    n_cams   = len(cameras)
    cam_ids  = list(cameras.keys())
    offset   = 0

    new_cameras = {}
    for cid in cam_ids:
        rvec = params[offset:offset+3];  offset += 3
        tvec = params[offset:offset+3];  offset += 3
        cam  = cameras[cid]
        from reconstruction import Camera   # local import avoids circular deps
        new_cameras[cid] = Camera(
            image_id   = cam.image_id,
            image_name = cam.image_name,
            R          = rvec_to_mat(rvec),
            t          = tvec.reshape(3,1),
            K          = cam.K,
        )

    new_tracks = []
    for track in tracks:
        from reconstruction import Track
        pt = params[offset:offset+3]; offset += 3
        new_tracks.append(Track(
            point3d      = pt,
            color        = track.color,
            observations = dict(track.observations),
        ))

    return new_cameras, new_tracks


# ─────────────────────────────────────────
#  Residual function
# ─────────────────────────────────────────

def _build_residuals(params, cameras, tracks, observations):
    """
    Compute reprojection residuals for every 2-D observation.

    observations : list of (cam_idx, track_idx, u_obs, v_obs)
                   — built once before optimisation starts.
    cameras_list : ordered list matching packed camera order.
    """
    n_cams      = len(cameras)
    cam_ids     = list(cameras.keys())
    residuals   = []

    for cam_idx, track_idx, u_obs, v_obs in observations:
        # — Unpack camera —
        base  = cam_idx * 6
        rvec  = params[base:base+3]
        tvec  = params[base+3:base+6]
        R     = rvec_to_mat(rvec)

        # — Unpack 3-D point —
        pt_base = n_cams * 6 + track_idx * 3
        X       = params[pt_base:pt_base+3]

        # — Grab K for this camera —
        K = cameras[cam_ids[cam_idx]].K

        # — Project —
        Xc    = R @ X + tvec               # camera-space
        if Xc[2] <= 0:                     # behind camera → large penalty
            residuals.extend([1e4, 1e4])
            continue

        u_proj = K[0,0] * Xc[0] / Xc[2] + K[0,2]
        v_proj = K[1,1] * Xc[1] / Xc[2] + K[1,2]

        residuals.append(u_proj - u_obs)
        residuals.append(v_proj - v_obs)

    return np.array(residuals)


# ─────────────────────────────────────────
#  Sparsity pattern (critical for speed)
# ─────────────────────────────────────────

def build_sparsity(n_cams: int, n_points: int, observations: list):
    """
    Build the Jacobian sparsity matrix.

    Each observation produces 2 residuals and touches:
      - 6 camera parameters
      - 3 point  parameters

    Without this, SciPy would use dense finite-differences and
    the problem would be ~100× slower.
    """
    n_residuals = 2 * len(observations)
    n_params    = n_cams * 6 + n_points * 3

    A = lil_matrix((n_residuals, n_params), dtype=int)

    for i, (cam_idx, pt_idx, *_) in enumerate(observations):
        row   = 2 * i
        # camera block
        c_off = cam_idx * 6
        A[row,   c_off:c_off+6] = 1
        A[row+1, c_off:c_off+6] = 1
        # point block
        p_off = n_cams * 6 + pt_idx * 3
        A[row,   p_off:p_off+3] = 1
        A[row+1, p_off:p_off+3] = 1

    return A.tocsr()


# ─────────────────────────────────────────
#  Main Bundle Adjuster
# ─────────────────────────────────────────

class BundleAdjuster:
    """
    Sparse Levenberg-Marquardt bundle adjustment
    using SciPy's least_squares with Jacobian sparsity.

    For larger scenes (500+ cameras) swap the backend to
    pycolmap.bundle_adjustment() which calls Ceres (C++).
    """

    def __init__(
        self,
        max_iterations:   int   = 200,
        function_tolerance: float = 1e-6,
        loss_function:    str   = 'huber',  # 'linear' | 'huber' | 'cauchy'
        verbose:          bool  = True,
    ):
        self.max_iter   = max_iterations
        self.ftol       = function_tolerance
        self.loss       = loss_function
        self.verbose    = verbose

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, cameras: dict, tracks: list) -> tuple[dict, list, dict]:
        """
        Run bundle adjustment.

        Parameters
        ----------
        cameras : dict[image_id → Camera]
        tracks  : list[Track]

        Returns
        -------
        cameras_opt : optimised cameras
        tracks_opt  : optimised tracks
        report      : dict with cost_before, cost_after, n_iterations, success
        """

        # — Build flat observation list —
        cam_id_to_idx = {cid: i for i, cid in enumerate(cameras.keys())}
        observations  = []

        for ti, track in enumerate(tracks):
            for img_id, (u, v) in track.observations.items():
                if img_id not in cam_id_to_idx:
                    continue
                observations.append(
                    (cam_id_to_idx[img_id], ti, u, v)
                )

        if len(observations) == 0:
            return cameras, tracks, {"success": False, "reason": "no observations"}

        # — Pack parameters —
        x0 = pack_params(cameras, tracks)

        # — Sparsity —
        A  = build_sparsity(len(cameras), len(tracks), observations)

        # — Cost before —
        res_before  = _build_residuals(x0, cameras, tracks, observations)
        cost_before = 0.5 * float(np.dot(res_before, res_before))

        if self.verbose:
            print(f"[BA] Observations : {len(observations)}")
            print(f"[BA] Parameters   : {len(x0)}")
            print(f"[BA] Cost (before): {cost_before:.4f}")

        # — Optimise —
        result = least_squares(
            fun          = _build_residuals,
            x0           = x0,
            args         = (cameras, tracks, observations),
            method       = 'trf',           # Trust Region Reflective — best for BA
            jac_sparsity = A,
            loss         = self.loss,
            f_scale      = 1.0,             # scale for robust loss (pixels)
            max_nfev     = self.max_iter,
            ftol         = self.ftol,
            xtol         = 1e-8,
            gtol         = 1e-8,
            verbose      = 2 if self.verbose else 0,
        )

        # — Unpack result —
        cameras_opt, tracks_opt = unpack_params(result.x, cameras, tracks)

        cost_after = 0.5 * float(np.dot(result.fun, result.fun))

        if self.verbose:
            print(f"[BA] Cost (after) : {cost_after:.4f}")
            print(f"[BA] Reduction    : {100*(1 - cost_after/cost_before):.1f}%")
            print(f"[BA] Iterations   : {result.nfev}")
            print(f"[BA] Success      : {result.success}")

        report = {
            "cost_before":   cost_before,
            "cost_after":    cost_after,
            "n_iterations":  result.nfev,
            "success":       result.success,
            "message":       result.message,
        }

        return cameras_opt, tracks_opt, report

    # ── Post-BA outlier pruning ───────────────────────────────────────────────

    def filter_outliers(
        self,
        cameras:   dict,
        tracks:    list,
        threshold: float = 2.0,     # pixels
    ) -> tuple[list, int]:
        """
        After BA, remove any track whose mean reprojection error
        across all observations exceeds `threshold` pixels.

        Returns (filtered_tracks, n_removed).
        """
        from reconstruction import Camera   # avoid circular import

        keep    = []
        removed = 0

        for track in tracks:
            errors = []
            for img_id, (u_obs, v_obs) in track.observations.items():
                cam = cameras.get(img_id)
                if cam is None:
                    continue
                Xc    = cam.R @ track.point3d + cam.t.ravel()
                if Xc[2] <= 0:
                    errors.append(1e6)
                    continue
                u_p   = cam.K[0,0] * Xc[0] / Xc[2] + cam.K[0,2]
                v_p   = cam.K[1,1] * Xc[1] / Xc[2] + cam.K[1,2]
                errors.append(np.sqrt((u_p-u_obs)**2 + (v_p-v_obs)**2))

            if errors and np.mean(errors) <= threshold:
                keep.append(track)
            else:
                removed += 1

        return keep, removed


# ─────────────────────────────────────────
#  Optional: drop-in COLMAP backend
#  (swap in for large scenes, 500+ cameras)
# ─────────────────────────────────────────

def run_colmap_ba(reconstruction):
    """
    Use pycolmap's C++ Ceres-backed bundle adjustment.
    Drop-in replacement for BundleAdjuster.run() on large scenes.

    Requires:  pip install pycolmap
    """
    try:
        import pycolmap
    except ImportError:
        raise RuntimeError("pip install pycolmap  to use the COLMAP BA backend")

    options = pycolmap.BundleAdjustmentOptions()
    options.solver_options.max_num_iterations  = 100
    options.solver_options.function_tolerance  = 1e-6
    options.refine_focal_length                = True
    options.refine_principal_point             = False
    options.refine_extra_params                = True   # distortion coefficients

    pycolmap.bundle_adjustment(reconstruction, options)
    return reconstruction