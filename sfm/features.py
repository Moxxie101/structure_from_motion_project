import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, method='sift', max_features=8000):
        if method == 'sift':
            self.detector = cv2.SIFT_create(nfeatures=max_features)
        elif method == 'orb':
            self.detector = cv2.ORB_create(nFeatures=max_features)

    def detect_and_compute(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None:
            return [], None
        return keypoints, descriptors  