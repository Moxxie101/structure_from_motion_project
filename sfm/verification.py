class GeometricVerifier:
    def verify(self, kp1, kp2, matches, min_inliers=15):
        if len(matches) < min_inliers:
            return None, []

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Fundamental matrix via RANSAC — encodes the epipolar geometry
        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            cv2.FM_RANSAC,
            ransacReprojThreshold=1.0,  # pixel error tolerance
            confidence=0.999
        )

        inliers = [matches[i] for i in range(len(matches)) if mask[i]]
        if len(inliers) < min_inliers:
            return None, []

        return F, inliers