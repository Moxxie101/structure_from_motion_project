import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────

@dataclass
class Camera:
    """Represents a registered camera / image in the reconstruction."""
    image_id:   int
    image_name: str
    R:          np.ndarray          # (3,3) rotation matrix     — world-to-camera
    t:          np.ndarray          # (3,1) translation vector  — world-to-camera
    K:          np.ndarray          # (3,3) intrinsic matrix
    inlier_ids: list = field(default_factory=list)  # indices into Track list

    @property
    def P(self) -> np.ndarray:
        """Projection matrix P = K @ [R | t]  (3x4)."""
        return self.K @ np.hstack([self.R, self.t])

    @property
    def center(self) -> np.ndarray:
        """Camera center in world coords  C = -R^T t."""
        return (-self.R.T @ self.t).ravel()


@dataclass
class Track:
    """A 3-D point and all 2-D observations that produced it."""
    point3d:      np.ndarray                    # (3,) XYZ in world frame
    color:        np.ndarray                    # (3,) RGB  0-255
    observations: dict = field(default_factory=dict)
    # observations = { image_id: (u, v) }  pixel that saw this point


# ─────────────────────────────────────────
#  Reconstruction Class
# ─────────────────────────────────────────

class Reconstruction:
    """
    Incremental Structure-from-Motion reconstruction.

    Typical call order
    ------------------
    1.  recon = Reconstruction(K)
    2.  recon.initialize(img0, img1, kp0, kp1, matches)   # seed two-view
    3.  for each new image:
            recon.register_image(img_id, name, kp, matches_to_existing)
            recon.triangulate_new_tracks(...)
            recon.filter_tracks()
    4.  Hand recon.cameras / recon.tracks to BundleAdjuster
    """

    def __init__(self, K: np.ndarray):
        self.K:       np.ndarray      = K
        self.cameras: dict[int, Camera] = {}   # image_id → Camera
        self.tracks:  list[Track]       = []

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _to_points(self, keypoints, matches, side: str) -> np.ndarray:
        """Pull (u,v) arrays from a keypoint list using a match list."""
        idx = 'queryIdx' if side == 'query' else 'trainIdx'
        return np.float32([keypoints[getattr(m, idx)].pt for m in matches])

    # ── Step 1 · Two-view initialisation ─────────────────────────────────────

    def initialize(
        self,
        id0: int, name0: str, kp0,
        id1: int, name1: str, kp1,
        matches,
        image0: np.ndarray,
        image1: np.ndarray,
    ) -> bool:
        """
        Bootstrap the reconstruction from the best two-view pair.

        Returns True on success, False if geometry is degenerate
        (e.g. pure rotation, insufficient parallax).
        """
        pts0 = self._to_points(kp0, matches, 'query')
        pts1 = self._to_points(kp1, matches, 'train')

        # — Essential matrix + pose recovery —
        E, mask_e = cv2.findEssentialMat(
            pts0, pts1, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if E is None:
            return False

        inlier_matches = [m for m, ok in zip(matches, mask_e.ravel()) if ok]
        pts0_in = self._to_points(kp0, inlier_matches, 'query')
        pts1_in = self._to_points(kp1, inlier_matches, 'train')

        n_inliers, R, t, mask_p = cv2.recoverPose(E, pts0_in, pts1_in, self.K)

        if n_inliers < 30:          # not enough geometric agreement
            return False

        # — Register both cameras —
        # Camera 0 sits at the world origin
        cam0 = Camera(id0, name0,
                      R=np.eye(3), t=np.zeros((3,1)), K=self.K)
        cam1 = Camera(id1, name1, R=R, t=t, K=self.K)
        self.cameras[id0] = cam0
        self.cameras[id1] = cam1

        # — Triangulate initial point cloud —
        good_mask = mask_p.ravel().astype(bool)
        pts0_final = pts0_in[good_mask]
        pts1_final = pts1_in[good_mask]
        good_matches = [inlier_matches[i]
                        for i, ok in enumerate(good_mask) if ok]

        self._triangulate_and_store(
            cam0, cam1,
            pts0_final, pts1_final,
            good_matches,
            kp0, kp1,
            image0, image1,
        )
        return True

    # ── Step 2 · Register a new image via PnP ────────────────────────────────

    def register_image(
        self,
        image_id:   int,
        image_name: str,
        keypoints,
        matches_2d3d: list,     # list of (kp_index, track_index) pairs
    ) -> bool:
        """
        Locate a new camera using known 3-D tracks (PnP + RANSAC).

        matches_2d3d must be pre-built by the caller by cross-referencing
        track observations with the new image's keypoints.

        Returns True on success.
        """
        if len(matches_2d3d) < 6:   # absolute minimum for PnP
            return False

        pts3d = np.float32(
            [self.tracks[ti].point3d for _, ti in matches_2d3d])
        pts2d = np.float32(
            [keypoints[ki].pt        for ki, _ in matches_2d3d])

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d,
            self.K, None,
            iterationsCount=1000,
            reprojectionError=2.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None or len(inliers) < 12:
            return False

        R, _ = cv2.Rodrigues(rvec)
        cam  = Camera(image_id, image_name, R=R, t=tvec, K=self.K)
        self.cameras[image_id] = cam

        # Record which tracks this camera observes
        for idx in inliers.ravel():
            ki, ti = matches_2d3d[idx]
            self.tracks[ti].observations[image_id] = keypoints[ki].pt
            cam.inlier_ids.append(ti)

        return True

    # ── Step 3 · Triangulate new tracks between two registered cameras ────────

    def triangulate_new_tracks(
        self,
        id_a: int, kp_a,
        id_b: int, kp_b,
        matches,
        image_a: np.ndarray,
        image_b: np.ndarray,
    ) -> int:
        """
        Triangulate matched points between two already-registered cameras
        and add them as new tracks.  Returns number of tracks added.
        """
        cam_a = self.cameras.get(id_a)
        cam_b = self.cameras.get(id_b)
        if cam_a is None or cam_b is None:
            return 0

        pts_a = self._to_points(kp_a, matches, 'query')
        pts_b = self._to_points(kp_b, matches, 'train')

        n_added = self._triangulate_and_store(
            cam_a, cam_b, pts_a, pts_b, matches, kp_a, kp_b, image_a, image_b
        )
        return n_added

    def _triangulate_and_store(
        self,
        cam_a: Camera, cam_b: Camera,
        pts_a: np.ndarray, pts_b: np.ndarray,
        matches,
        kp_a, kp_b,
        image_a: np.ndarray,
        image_b: np.ndarray,
    ) -> int:
        """Low-level triangulation + cheirality check + track storage."""
        pts4d   = cv2.triangulatePoints(cam_a.P, cam_b.P, pts_a.T, pts_b.T)
        pts3d   = (pts4d[:3] / pts4d[3]).T          # (N,3) Euclidean

        n_added = 0
        for i, (pt3d, m) in enumerate(zip(pts3d, matches)):
            # — Cheirality: point must be in front of both cameras —
            if not self._positive_depth(pt3d, cam_a):
                continue
            if not self._positive_depth(pt3d, cam_b):
                continue

            # — Reprojection error guard —
            err_a = self._reproj_error(pt3d, kp_a[m.queryIdx].pt, cam_a)
            err_b = self._reproj_error(pt3d, kp_b[m.trainIdx].pt, cam_b)
            if err_a > 4.0 or err_b > 4.0:         # pixels
                continue

            # — Sample colour from image_a —
            u, v  = kp_a[m.queryIdx].pt
            color = image_a[int(v), int(u)][::-1]   # BGR → RGB

            track = Track(
                point3d=pt3d,
                color=color.copy(),
                observations={
                    cam_a.image_id: kp_a[m.queryIdx].pt,
                    cam_b.image_id: kp_b[m.trainIdx].pt,
                },
            )
            self.tracks.append(track)
            n_added += 1

        return n_added

    # ── Step 4 · Outlier filtering ────────────────────────────────────────────

    def filter_tracks(
        self,
        max_reproj_error: float = 4.0,
        min_observations: int   = 2,
    ) -> int:
        """
        Remove tracks that:
        - reproject poorly in any observing camera, or
        - are seen in too few cameras (weak triangulation).

        Returns number of tracks removed.
        """
        keep   = []
        before = len(self.tracks)

        for track in self.tracks:
            if len(track.observations) < min_observations:
                continue

            bad = False
            for img_id, uv in track.observations.items():
                cam = self.cameras.get(img_id)
                if cam is None:
                    continue
                err = self._reproj_error(track.point3d, uv, cam)
                if err > max_reproj_error:
                    bad = True
                    break

            if not bad:
                keep.append(track)

        self.tracks = keep
        return before - len(keep)

    # ── Geometry utilities ───────────────────────────────────────────────────

    @staticmethod
    def _positive_depth(pt3d: np.ndarray, cam: Camera) -> bool:
        """Return True if the point is in front of the camera (z > 0)."""
        pt_cam = cam.R @ pt3d + cam.t.ravel()
        return float(pt_cam[2]) > 0.0

    @staticmethod
    def _reproj_error(pt3d: np.ndarray, uv_obs, cam: Camera) -> float:
        """Euclidean reprojection error in pixels."""
        pt_h   = np.append(pt3d, 1.0)
        proj   = cam.P @ pt_h
        proj  /= proj[2]
        return float(np.linalg.norm(proj[:2] - np.array(uv_obs)))

    # ── Accessors ────────────────────────────────────────────────────────────

    def point_cloud(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (Nx3 XYZ, Nx3 RGB) arrays for export or visualisation."""
        if not self.tracks:
            return np.empty((0,3)), np.empty((0,3))
        pts    = np.array([t.point3d for t in self.tracks])
        colors = np.array([t.color   for t in self.tracks])
        return pts, colors

    def camera_poses(self) -> list[dict]:
        """Return list of {image_id, center, R} for all registered cameras."""
        return [
            {"image_id": cam.image_id,
             "name":     cam.image_name,
             "center":   cam.center,
             "R":        cam.R}
            for cam in self.cameras.values()
        ]

    def summary(self) -> str:
        return (f"Cameras registered : {len(self.cameras)}\n"
                f"Tracks (3-D points): {len(self.tracks)}")