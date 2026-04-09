class FeatureMatcher:
    def __init__(self, method='flann'):
        if method == 'flann':
            index_params = dict(algorithm=1, trees=5)  # KD-tree
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.matcher = cv2.BFMatcher()

    def match(self, desc1, desc2, ratio_threshold=0.75):
        raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for m, n in raw_matches:
            if m.distance < ratio_threshold * n.distance:  # Lowe's ratio test
                good.append(m)
        return good