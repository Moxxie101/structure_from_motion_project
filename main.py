import argparse, cv2, os
import numpy as np
from sfm.loader import ImageLoader
from sfm.features import FeatureExtractor
from sfm.matching import FeatureMatcher
from sfm.verification import GeometricVerifier
from sfm.reconstruction import Reconstruction, Track

OUTPUT_ROOT = "output"
FOLDERS = ["features", "matching", "verification", "reconstruction"]

def make_output_dirs():
    for folder in FOLDERS:
        os.makedirs(os.path.join(OUTPUT_ROOT, folder), exist_ok=True)

def save(folder: str, filename: str, image):
    path = os.path.join(OUTPUT_ROOT, folder, filename)
    cv2.imwrite(path, image)
    print(f"  Saved -> {path}")

class SfMPipeline:
    def __init__(self, image_path, K=None, method='sift'):
        self.loader    = ImageLoader(image_path)
        self.extractor = FeatureExtractor(method=method)
        self.matcher   = FeatureMatcher(method='flann', descriptor_type=method)
        self.verifier  = GeometricVerifier()
        self.images    = []
        self.features  = []
        self.K         = K 

    def run(self):
        make_output_dirs()

        self.images = self.loader.load_images()
        print(f"Loaded {len(self.images)} images")

        print("\n[Features]")
        self.features = []
        for idx, img in enumerate(self.images):
            kps, descs = self.extractor.detect_and_compute(img)
            self.features.append((kps, descs))
            print(f"  Image {idx}: {len(kps)} keypoints")

            vis = cv2.drawKeypoints(
                img, kps, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            save("features", f"keypoints_{idx:03d}.jpg", vis)

        if self.K is None:
            self.K = self._estimate_K(self.images[0])

        recon = Reconstruction(self.K)

        print("\n[Matching]")
        all_matches = {}
        for i in range(len(self.images) - 1):
            kps1, desc1 = self.features[i]
            kps2, desc2 = self.features[i + 1]

            raw = self.matcher.match(desc1, desc2)
            print(f"  Pair {i}<>{i+1}: {len(raw)} raw matches")

            vis_raw = cv2.drawMatches(
                self.images[i], kps1,
                self.images[i+1], kps2,
                raw[:50], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            save("matching", f"matches_raw_{i:03d}_{i+1:03d}.jpg", vis_raw)

            print(f"\n[Verification] Pair {i}<>{i+1}")
            _, inliers = self.verifier.verify(kps1, kps2, raw)
            if not inliers:
                print(f"  Pair {i}<>{i+1}: verification failed, skipping")
                continue

            print(f"  Pair {i}<>{i+1}: {len(inliers)} inliers after RANSAC")
            all_matches[(i, i+1)] = inliers

            vis_ver = cv2.drawMatches(
                self.images[i], kps1,
                self.images[i+1], kps2,
                inliers[:50], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            save("verification", f"verified_{i:03d}_{i+1:03d}.jpg", vis_ver)

        if not all_matches:
            print("No valid pairs found. Exiting.")
            return recon

        print("\n[Reconstruction]")
        best_pair = max(all_matches, key=lambda k: len(all_matches[k]))
        i, j      = best_pair
        kps_i, _  = self.features[i]
        kps_j, _  = self.features[j]

        print(f"  Initializing from pair {i}<>{j} ({len(all_matches[best_pair])} inliers)")
        ok = recon.initialize(
            i, f"image_{i:03d}", kps_i,
            j, f"image_{j:03d}", kps_j,
            all_matches[best_pair],
            self.images[i], self.images[j],
        )
        if not ok:
            print("  Initialization failed — degenerate geometry.")
            return recon

        print(f"  {recon.summary()}")
        registered = {i, j}

        for new_id in range(len(self.images)):
            if new_id in registered:
                continue

            matches_2d3d = self._build_2d3d_matches(new_id, registered, recon)
            if not matches_2d3d:
                print(f"  Image {new_id}: no 2D-3D matches, skipping")
                continue

            ok = recon.register_image(
                new_id, f"image_{new_id:03d}",
                self.features[new_id][0],
                matches_2d3d,
            )
            if not ok:
                print(f"  Image {new_id}: PnP failed, skipping")
                continue

            registered.add(new_id)

            for reg_id in registered - {new_id}:
                pair_key = (min(reg_id, new_id), max(reg_id, new_id))
                if pair_key not in all_matches:
                    continue
                n = recon.triangulate_new_tracks(
                    reg_id, self.features[reg_id][0],
                    new_id, self.features[new_id][0],
                    all_matches[pair_key],
                    self.images[reg_id], self.images[new_id],
                )
                print(f"  Triangulated {n} new tracks from pair {reg_id}<>{new_id}")

            removed = recon.filter_tracks()
            print(f"  Image {new_id} registered | {recon.summary()} | Filtered {removed} tracks")

        pts, colors = recon.point_cloud()
        print(f"\nFinal {recon.summary()}")
        print(f"Point cloud: {len(pts)} points")
        self._save_point_cloud_preview(pts, colors)

        return recon

    def _estimate_K(self, image) -> 'np.ndarray':
        h, w = image.shape[:2]
        f    = max(h, w)
        return np.array([
            [f,  0,  w / 2],
            [0,  f,  h / 2],
            [0,  0,  1    ],
        ], dtype=float)

    def _build_2d3d_matches(self, new_id, registered, recon) -> list:
        matches_2d3d = []
        kps_new, desc_new = self.features[new_id]

        for reg_id in registered:
            kps_reg, desc_reg = self.features[reg_id]
            raw = self.matcher.match(desc_new, desc_reg)
            _, inliers = self.verifier.verify(kps_new, kps_reg, raw)

            obs_map = {}
            for ti, track in enumerate(recon.tracks):
                if reg_id in track.observations:
                    obs_map[track.observations[reg_id]] = ti

            for m in inliers:
                uv_reg = kps_reg[m.trainIdx].pt
                if uv_reg in obs_map:
                    matches_2d3d.append((m.queryIdx, obs_map[uv_reg]))

        return matches_2d3d

    def _save_point_cloud_preview(self, pts, colors):
        """Save a basic top-down XZ scatter of the point cloud as an image."""
        import numpy as np
        if len(pts) == 0:
            return

        canvas_size = 800
        canvas = 255 * __import__('numpy').ones((canvas_size, canvas_size, 3), dtype='uint8')

        xs, zs = pts[:, 0], pts[:, 2]
        margin = 0.05
        x_min, x_max = xs.min(), xs.max()
        z_min, z_max = zs.min(), zs.max()
        x_range = max(x_max - x_min, 1e-6)
        z_range = max(z_max - z_min, 1e-6)

        for pt, color in zip(pts, colors):
            x = int((pt[0] - x_min) / x_range * (1 - 2*margin) * canvas_size + margin * canvas_size)
            z = int((pt[2] - z_min) / z_range * (1 - 2*margin) * canvas_size + margin * canvas_size)
            bgr = (int(color[2]), int(color[1]), int(color[0]))
            cv2.circle(canvas, (x, z), 2, bgr, -1)

        save("reconstruction", "point_cloud_preview.jpg", canvas)


# loader = ImageLoader(args.path)
# images = loader.load_images()
# extractor = FeatureExtractor(method='sift')
# matcher = FeatureMatcher(method='flann', descriptor_type='sift')
# verifier = GeometricVerifier()
#
# features = []
# for img in images:
#     kps, descs = extractor.detect_and_compute(img)
#     features.append((kps, descs))
#     print(f"Found {len(kps)} keypoints")
# for img, (kps, _) in zip(images, features):
#     vis = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     cv2.imshow("Keypoints", vis)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()


# features = [extractor.detect_and_compute(img) for img in images]
# for i in range(len(images) - 1):
#     kps1, desc1 = features[i]
#     kps2, desc2 = features[i + 1]
#     matches = matcher.match(desc1, desc2)
#     print(f"Pair {i}<>{i+1}: {len(matches)} good matches")
# vis = cv2.drawMatches(images[0], features[0][0],
#                       images[1], features[1][0],
#                       matches[:50], None,
#                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.imshow("Matches", vis)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

parser = argparse.ArgumentParser(description="Structure from Motion pipeline")
parser.add_argument("--path", "-p", type=str, default=None,
                    help="Path to folder of images or a video file")
parser.add_argument("--method", "-m", type=str, default="sift",
                    choices=["sift", "orb"],
                    help="Feature extraction method (default: sift)")
args = parser.parse_args()

if args.path is None:
    print("No path provided. Use --path or -p to specify an image folder.")
    exit(1)

pipeline = SfMPipeline(args.path, method=args.method)
recon    = pipeline.run()