def recover_pose(self, kp1, kp2, matches, K):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    # R = rotation matrix (3x3), t = translation vector (3x1)
    return R, t

def register_new_image(self, pts3d, pts2d, K):
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, None,
        iterationsCount=1000,
        reprojectionError=2.0,
        confidence=0.999
    )
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec

def triangulate(self, P1, P2, pts1, pts2):
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T  # convert from homogeneous
    return pts3d