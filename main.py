import argparse, cv2
from sfm.loader import ImageLoader

parser = argparse.ArgumentParser(description="My script")
parser.add_argument("name")
parser.add_argument("--path", "-p", type=str, default=None)
args = parser.parse_args()

if args.path is None:
    print("no path provided")
    exit(1)

loader = ImageLoader(args.path)

# for image in loader.load_images():
    # print(image)