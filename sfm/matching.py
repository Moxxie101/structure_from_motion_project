import cv2

class FeatureMatcher:
    def __init__(self, method='flann', descriptor_type='sift'):
        if method == 'flann':
            if descriptor_type == 'orb':
                index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            else:
                index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif method == 'bf':
            norm = cv2.NORM_HAMMING if descriptor_type == 'orb' else cv2.NORM_L2
            self.matcher = cv2.BFMatcher(norm)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'flann' or 'bf'.")

    def match(self, desc1, desc2, ratio_threshold=0.75):
        if desc1 is None or desc2 is None:
            return []

        raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio_threshold * n.distance:
                    good.append(m)
        return good