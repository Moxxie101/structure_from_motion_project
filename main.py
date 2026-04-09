import argparse, cv2
from sfm.loader import ImageLoader
from sfm.features import FeatureExtractor
from sfm.matching import FeatureMatcher

parser = argparse.ArgumentParser(description="My script")
parser.add_argument("--path", "-p", type=str, default=None)
args = parser.parse_args()

if args.path is None:
    print("no path provided")
    exit(1)

loader = ImageLoader(args.path)
images = loader.load_images()
extractor = FeatureExtractor(method='sift')
matcher = FeatureMatcher(method='flann', descriptor_type='sift')


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


features = [extractor.detect_and_compute(img) for img in images]
for i in range(len(images) - 1):
    kps1, desc1 = features[i]
    kps2, desc2 = features[i + 1]
    matches = matcher.match(desc1, desc2)
    print(f"Pair {i}<>{i+1}: {len(matches)} good matches")
vis = cv2.drawMatches(images[0], features[0][0],
                      images[1], features[1][0],
                      matches[:50], None,
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Matches", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()



