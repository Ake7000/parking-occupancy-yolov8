# main_yolo.py
# 1) Calibrates the slot ROIs (parking spaces) from the orange slot numbers + white markings
# 2) Runs YOLO on the FULL FRAME
# 3) Maps each detected car's bounding box to the matching ROI
# 4) Prints the slot states to the terminal: free / occupied
#    and updates only when a state changes

import cv2
import json
import os
import time
from ultralytics import YOLO

# Try to import the calibration function from roi_detection
try:
    from roi_detection.compute_rois import compute_rois_and_save_from_frame
    HAS_CALIBRATION_FUNC = True
except ImportError:
    HAS_CALIBRATION_FUNC = False
    print("[WARN] Nu am gasit compute_rois_and_save_from_frame in roi_detection.compute_rois.")
    print("       Asigura-te ca ai rulat separat roi_detection/compute_rois.py ca sa generezi rois.json.")


ROIS_JSON_PATH = "rois.json"      # compute_rois.py saves the ROIs here
CAM_INDEX = 1                     # camera index of the overhead camera
FRAME_WIDTH = 2048
FRAME_HEIGHT = 1152

YOLO_WEIGHTS = "model/best.pt"    # fine-tuned detector weights
CONF_TH = 0.60                    # YOLO confidence threshold
DEBUG_DETECTIONS = False           # print every detection (debugging)


# ------------------------------------------------
# Utility: load ROIs from rois.json
# ------------------------------------------------
def load_rois(path=ROIS_JSON_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nu gasesc fisierul ROI: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    rois = data.get("rois", [])
    # sort by id to make sure they are in order 1..6
    rois = sorted(rois, key=lambda r: r.get("id", 0))
    return rois


# ------------------------------------------------
# Utility: ROI calibration from a frame
# ------------------------------------------------
def calibrate_rois_from_camera(cap, buffer_factor=1.15):
    """
    Shows a live preview and waits for SPACE to calibrate.
    Uses compute_rois_and_save_from_frame() when it is available.
    Returns the list of ROIs.
    """
    if not HAS_CALIBRATION_FUNC:
        # fallback: just load an existing rois.json
        print("[INFO] Nu exista compute_rois_and_save_from_frame; sar peste calibrare live.")
        return load_rois()

    print("=== CALIBRARE ROI ===")
    print("Apasa SPACE pentru a captura un frame pentru calibrare, 'q' pentru iesire.")

    frame_for_calib = None

    while True:
        ret, frm = cap.read()
        if not ret:
            print("[Eroare] Nu pot citi frame de la camera in timpul calibrarii.")
            continue

        cv2.imshow("Calibrare ROI - live", frm)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            frame_for_calib = frm.copy()
            print("[INFO] Cadru pentru calibrare capturat.")
            break
        elif key == ord('q'):
            print("[INFO] Ai apasat 'q' in timpul calibrarii. Ies.")
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    cv2.destroyWindow("Calibrare ROI - live")

    # Use the helper from compute_rois.py
    print("[INFO] Calculez ROI-urile pe baza cadrului capturat...")
    rois = compute_rois_and_save_from_frame(frame_for_calib, buffer_factor=buffer_factor,
                                            save_path=ROIS_JSON_PATH)

    print("[INFO] ROI-uri calibrate si salvate in rois.json:")
    for r in rois:
        print(f"  Loc {r['id']}: x={r['x']}, y={r['y']}, w={r['w']}, h={r['h']}")

    return rois


# ------------------------------------------------
# Function: map YOLO detections -> parking slots
# ------------------------------------------------
def is_roi_occupied(roi, detections, car_like_classes):
    """
    roi: dict {id, x, y, w, h}
    detections: list of tuples (cls_name, conf, x1, y1, x2, y2)
    car_like_classes: set of accepted class names
    Returns True if the centre of any accepted bounding box falls inside the ROI.
    """
    rx, ry, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
    rx2, ry2 = rx + rw, ry + rh

    for cls_name, conf, x1, y1, x2, y2 in detections:
        if cls_name not in car_like_classes:
            continue

        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        if (rx <= cx <= rx2) and (ry <= cy <= ry2):
            return True

    return False


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    # 1) Open the camera
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[Eroare] Nu pot deschide camera cu index {CAM_INDEX}")
        return

    # 2) ROI calibration (or load from file)
    rois = calibrate_rois_from_camera(cap, buffer_factor=1.15)

    # 3) Load the YOLO model
    print(f"[INFO] Incarc modelul YOLO din {YOLO_WEIGHTS}...")
    model = YOLO(YOLO_WEIGHTS)
    names = model.names  # dict: id -> class name
    print("[YOLO] Clase in model:", names)

    # Treat every class the model predicts as a car
    car_like_classes = set(names.values())
    print("[YOLO] Clase tratate ca 'masina':", car_like_classes)

    # 4) Main loop: read frames, run YOLO, map detections to ROIs
    last_statuses = None
    print("\n=== Pornesc monitorizarea parcarii (q pentru iesire) ===")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Eroare] Nu pot citi frame de la camera.")
            break

        # Run YOLO on the full frame
        results = model(frame, conf=CONF_TH, verbose=False)
        detections = []

        if len(results) > 0:
            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    detections.append((cls_name, conf, x1, y1, x2, y2))

        if DEBUG_DETECTIONS:
            print("\n[DEBUG] Detectii YOLO in acest frame:")
            if not detections:
                print("  (niciuna)")
            else:
                for cls_name, conf, x1, y1, x2, y2 in detections:
                    print(f"  {cls_name} conf={conf:.3f} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

        # 5) Decide the state of each slot
        statuses = []
        for roi in rois:
            occupied = is_roi_occupied(roi, detections, car_like_classes)
            statuses.append("occupied" if occupied else "free")

        # 6) Print to the terminal only if something changed
        if last_statuses is None or statuses != last_statuses:
            msg = "[update] " + " ".join(
                f"{roi['id']}={statuses[i]}"
                for i, roi in enumerate(rois)
            )
            print(msg)
            last_statuses = statuses

        # 7) Draw the overlay for visual debugging
        vis = frame.copy()

        # draw the YOLO bounding boxes
        for cls_name, conf, x1, y1, x2, y2 in detections:
            color = (0, 0, 255)  # red for detections
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(vis, f"{cls_name} {conf:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # draw the ROIs and their state
        for i, roi in enumerate(rois):
            rx, ry, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
            slot_id = roi["id"]
            status = statuses[i]

            if status == "occupied":
                color = (0, 0, 255)  # red
            else:
                color = (0, 255, 0)  # green

            cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), color, 2)
            cv2.putText(vis, f"{slot_id}:{status}",
                        (rx + 5, ry + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Parking YOLO - full frame", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Ai apasat 'q'. Iesire.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
