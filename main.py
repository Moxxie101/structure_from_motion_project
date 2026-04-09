import argparse, cv2
from sfm.loader import ImageLoader
from sfm.features import FeatureExtractor

parser = argparse.ArgumentParser(description="My script")
parser.add_argument("--path", "-p", type=str, default=None)
args = parser.parse_args()

if args.path is None:
    print("no path provided")
    exit(1)

loader = ImageLoader(args.path)
images = loader.load_images()
extractor = FeatureExtractor(method='sift')
features = []
for img in images:
    kps, descs = extractor.detect_and_compute(img)
    features.append((kps, descs))
    print(f"Found {len(kps)} keypoints")
for img, (kps, _) in zip(images, features):
    vis = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints", vis)
    cv2.waitKey(0)

cv2.destroyAllWindows()
