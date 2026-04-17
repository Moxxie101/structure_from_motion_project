import argparse
from Image_stitch.stitcher import stitch

parser = argparse.ArgumentParser(description="Structure from Motion pipeline")
parser.add_argument("--path", "-p", type=str, default=None,
                    help="Path to folder of images or a video file")
parser.add_argument("--mode", "-m", type=str, default="panorama",
                    choices=["panorama", "scans"],
                    help="Feature extraction method [panorama or scans] (default: panorama)")
args = parser.parse_args()

if args.path is None:
    print("No path provided. Use --path or -p to specify an image folder.")
    exit(1)

stitch(args.mode,args.path)
