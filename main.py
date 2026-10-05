# main.py
# Main loop: reads the camera, loads rois.json,
# uses SimpleSlotClassifier and prints the slot states to the terminal.

import cv2
import json
import time

from inference.simple_classifier import SimpleSlotClassifier


def load_rois(json_path="rois.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["rois"]


def format_status(statuses):
    """
    Takes a list such as ['free', 'occupied', ...] and
    returns a readable status string for printing.
    """
    parts = []
    for i, st in enumerate(statuses, start=1):
        parts.append(f"{i}:{'L' if st=='free' else 'O'}")
    return " | ".join(parts)


def main():
    rois = load_rois("rois.json")
    num_slots = len(rois)
    print(f"Am incarcat {num_slots} ROI-uri din rois.json")

    cam_index = 0  # camera index (change if needed)
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print(f"Eroare: nu pot deschide camera cu index {cam_index}")
        return

    classifier = SimpleSlotClassifier(delta_s=15.0, ema_alpha=0.05)

    print("=== Faza 1: CALIBRARE (parcarea TREBUIE sa fie goala) ===")
    print("Pune macheta in fata camerei fara masini pe locuri.")
    print("Apasa 'c' pentru calibrare, 'q' pentru iesire.")

    frame = None
    while True:
        ret, frm = cap.read()
        if not ret:
            print("Nu pot citi frame de la camera.")
            continue

        # Draw the ROIs to show what is being calibrated
        debug = frm.copy()
        for roi in rois:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Calibrare - parcarea goala", debug)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            frame = frm.copy()
            classifier.calibrate(frame, rois)
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    print("=== Faza 2: MONITORIZARE LIVE ===")
    print("Apasa 'q' pentru iesire.")

    last_statuses = [None] * num_slots
    last_print_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nu pot citi frame de la camera.")
            continue

        statuses = classifier.predict(frame, rois)  # 'free' / 'occupied'

        # Print only if something changed since the previous frame
        if statuses != last_statuses:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] Update status: {format_status(statuses)}")
            last_statuses = statuses[:]

        # Optional: also show a window with the ROIs colour-coded by state
        debug = frame.copy()
        for i, roi in enumerate(rois):
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            color = (0, 255, 0) if statuses[i] == "free" else (0, 0, 255)
            cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
            cv2.putText(debug, str(i+1), (x + 5, y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Monitorizare parcari", debug)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
