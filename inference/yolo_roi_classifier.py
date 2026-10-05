# inference/yolo_roi_classifier.py
# Per-ROI classification: decides whether a car is present in each slot crop using YOLO (alternative approach, not used by main_yolo.py).

import cv2
from ultralytics import YOLO
import numpy as np


class YoloRoiCarClassifier:
    def __init__(self,
                 weights: str = "model/best.pt",
                 car_like_classes=None,
                 conf_th: float = 0.60,
                 debug: bool = True):
        """
        weights: path to the YOLO weights (e.g. 'model/best.pt')
        car_like_classes:
            - None => treat EVERY class of the model as a car (all of model.names)
            - otherwise => set of accepted class names
        conf_th: minimum confidence for detections
        debug: if True, print the detections found in each ROI
        """
        self.model = YOLO(weights)
        self.conf_th = conf_th
        self.debug = debug

        # class names of the model
        self.names = self.model.names  # dict id -> name
        print("[YOLO] Clase in model:", self.names)

        if car_like_classes is None:
            # Custom model: treat every class of the model as a car
            # (i.e. all names in model.names)
            self.car_like_classes = set(self.names.values())
        else:
            self.car_like_classes = set(car_like_classes)

        print("[YOLO] Clase tratate ca 'masina':", self.car_like_classes)

    def roi_has_car(self, frame_bgr, roi: dict) -> bool:
        """
        frame_bgr: full image (BGR)
        roi: dict with keys 'x', 'y', 'w', 'h'
        Returns True if YOLO detects an accepted class inside the ROI.
        """
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        patch = frame_bgr[y:y + h, x:x + w]

        if patch.size == 0:
            if self.debug:
                print(f"[ROI {roi['id']}] patch gol, sar peste.")
            return False

        # Run YOLO on the crop (BGR). YOLO resizes it to imgsz internally.
        results = self.model(patch, conf=self.conf_th, verbose=False)

        if len(results) == 0:
            if self.debug:
                print(f"[ROI {roi['id']}] niciun rezultat YOLO.")
            return False

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            if self.debug:
                print(f"[ROI {roi['id']}] niciun bounding box.")
            return False

        has_car = False
        if self.debug:
            print(f"[ROI {roi['id']}] {len(res.boxes)} bounding box-uri gasite:")

        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.names.get(cls_id, str(cls_id))
            conf = float(box.conf[0].item())

            if self.debug:
                print(f"   - cls_id={cls_id}, cls_name={cls_name}, conf={conf:.3f}")

            if cls_name in self.car_like_classes and conf >= self.conf_th:
                has_car = True

        return has_car

    def predict_slots(self, frame_bgr, rois):
        """
        rois: list of dicts {id, x, y, w, h}
        Returns a list ['free'/'occupied', ...] in the same order.
        """
        statuses = []
        for roi in rois:
            has_car = self.roi_has_car(frame_bgr, roi)
            statuses.append("occupied" if has_car else "free")
        return statuses
