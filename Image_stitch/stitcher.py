import cv2
import os
import glob


def stitch(mode: str, path: str):
    """
    Stitch images from a folder or frames from a video file.

    Args:
        mode: Stitching mode - 'panorama' or 'scans'
        path: Path to a folder of images or a video file
    """

    images = []
    source_name = os.path.splitext(os.path.basename(path.rstrip("/\\")))[0]

    if os.path.isdir(path):
        extensions = ("*.jpg", "*.jpeg", "*.png")
        filepaths = []
        for ext in extensions:
            filepaths.extend(glob.glob(os.path.join(path, ext)))
            filepaths.extend(glob.glob(os.path.join(path, ext.upper())))
        filepaths = sorted(set(filepaths))

        if not filepaths:
            print(f"[ERROR] No images found in folder: {path}")
            return

        print(f"[INFO] Found {len(filepaths)} image(s) in '{path}'")
        for fp in filepaths:
            img = cv2.imread(fp)
            if img is not None:
                images.append(img)
            else:
                print(f"[WARNING] Could not read image: {fp}")

    elif os.path.isfile(path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open file: {path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = 20
        step = max(1, total_frames // max_frames)
        frame_idx = 0

        print(f"[INFO] Sampling frames from video '{path}' (every {step} frame(s))")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                images.append(frame)
            frame_idx += 1
        cap.release()

    else:
        print(f"[ERROR] Path does not exist: {path}")
        return

    if len(images) < 2:
        print("[ERROR] Need at least 2 images to stitch.")
        return

    print(f"[INFO] Stitching {len(images)} image(s) in '{mode}' mode...")

    stitch_mode = (
        cv2.Stitcher_PANORAMA if mode == "panorama" else cv2.Stitcher_SCANS
    )
    stitcher = cv2.Stitcher.create(stitch_mode)

    status, result = stitcher.stitch(images)

    status_messages = {
        cv2.Stitcher_OK: "OK",
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed",
    }

    if status != cv2.Stitcher_OK:
        reason = status_messages.get(status, f"Unknown error (code {status})")
        print(f"[ERROR] Stitching failed: {reason}")
        return

    output_dir = os.path.dirname(os.path.abspath(path.rstrip("/\\")))
    output_filename = f"stitched_{source_name}.jpg"
    output_path = os.path.join(output_dir, output_filename)

    cv2.imwrite(output_path, result)
    print(f"[INFO] Stitched image saved to: {output_path}")