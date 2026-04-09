import cv2, os

class ImageLoader:
    def __init__(self, source_path, max_dim=1600):
        self.source_path = source_path
        self.max_dim = max_dim  # downsample for speed without killing quality

    def load_images(self):
        images = []
        if self._is_video():
            images = self._extract_frames()
        else:
            for f in sorted(os.listdir(self.source_path)):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img = cv2.imread(os.path.join(self.source_path, f))
                    images.append(self._resize(img))
        return images

    def _resize(self, img):
        h, w = img.shape[:2]
        scale = self.max_dim / max(h, w)
        if scale < 1.0:
            return cv2.resize(img, (int(w*scale), int(h*scale)))
        return img

    def _is_video(self):
        return self.source_path.lower().endswith(('.mp4', '.mov', '.avi'))

    def _extract_frames(self, fps_target=2):
        # Extract at reduced FPS to avoid redundant near-identical frames
        cap = cv2.VideoCapture(self.source_path)
        frames, video_fps = [], cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(video_fps / fps_target))
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if i % interval == 0:
                frames.append(self._resize(frame))
            i += 1
        return frames