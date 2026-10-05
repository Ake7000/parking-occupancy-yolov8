# inference/yolo_full_frame_debug.py
# Debug: shows what YOLO detects on the full camera frame.

import cv2
from ultralytics import YOLO

def main():
    model = YOLO("model/best.pt")  # fine-tuned detector

    cam_index = 0  # camera index
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print(f"Eroare: nu pot deschide camera cu index {cam_index}")
        return

    print("YOLO full-frame debug pornit. Apasa 'q' pentru iesire.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nu pot citi frame de la camera.")
            continue

        # run YOLO on the full frame with a relaxed threshold
        results = model(frame, conf=0.15, verbose=False)
        r = results[0]

        names = model.names

        annotated = frame.copy()

        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = names.get(cls_id, str(cls_id))

                # draw the bounding box
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.rectangle(annotated, pt1, pt2, (0, 255, 0), 2)

                label = f"{cls_name} {conf:.2f}"
                cv2.putText(annotated, label, (pt1[0], pt1[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLO full-frame debug", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
