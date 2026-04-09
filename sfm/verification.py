import cv2
import numpy as np

class GeometricVerifier:
    def verify(self, kp1, kp2, matches, min_inliers=15):
        if len(matches) < min_inliers:
            return None, []

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            cv2.FM_RANSAC,
            ransacReprojThreshold=1.0,
            confidence=0.999
        )

        if F is None or F.shape != (3, 3):
            return None, []

        if mask is None:
            return None, []

        inliers = [matches[i] for i in range(len(matches)) if mask[i][0]]
        if len(inliers) < min_inliers:
            return None, []

        return F, inliers