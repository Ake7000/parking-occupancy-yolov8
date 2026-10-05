# inference/simple_classifier.py
# Simple classifier: calibration on the empty lot + detection based on saturation (S channel of HSV).

import cv2
import numpy as np


class SimpleSlotClassifier:
    def __init__(self, delta_s=15.0, ema_alpha=0.05):
        """
        delta_s: margin above the saturation baseline for a slot to count as 'occupied'
        ema_alpha: how fast the baseline adapts (EMA) while the slot is free
        """
        self.delta_s = delta_s
        self.ema_alpha = ema_alpha
        self.baseline_s = None  # list of mean S values, one per ROI

    def _roi_mean_saturation(self, frame_bgr, roi):
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        patch = frame_bgr[y:y+h, x:x+w]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        return float(np.mean(s_channel))

    def calibrate(self, frame_bgr, rois):
        """
        Calibration on the empty lot: stores the mean S value of each slot.
        Assumes there are no cars in the lot during calibration.
        """
        self.baseline_s = []
        for roi in rois:
            mean_s = self._roi_mean_saturation(frame_bgr, roi)
            self.baseline_s.append(mean_s)
        print("Calibrare facuta. Baseline S per slot:", self.baseline_s)

    def predict(self, frame_bgr, rois):
        """
        Returns one label per ROI:
          'free' or 'occupied'
        """
        if self.baseline_s is None:
            raise RuntimeError("Classifier nu este calibrat. Apeleaza calibrate() mai intai.")

        statuses = []
        for i, roi in enumerate(rois):
            mean_s = self._roi_mean_saturation(frame_bgr, roi)
            base = self.baseline_s[i]

            # if saturation rose well above the baseline => probably a car
            if mean_s > base + self.delta_s:
                status = "occupied"
            else:
                status = "free"
                # if the slot is free, slowly update the baseline (lighting can change)
                new_base = (1.0 - self.ema_alpha) * base + self.ema_alpha * mean_s
                self.baseline_s[i] = new_base

            statuses.append(status)

        return statuses
