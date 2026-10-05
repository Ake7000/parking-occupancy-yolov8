# capture_dataset.py
# Capture images from the parking camera to build a Hot Wheels dataset.
# Press 's' to save a frame, 'q' to quit.

import cv2
import os

def main():
    output_dir = "data_hotwheels/raw"
    os.makedirs(output_dir, exist_ok=True)

    cam_index = 0  # camera index; change it if your camera uses a different index
    cap = cv2.VideoCapture(cam_index)

    # Try mid resolution (16:9)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print(f"Error: cannot open camera index {cam_index}")
        return

    # find next index for filenames
    existing = [
        f for f in os.listdir(output_dir)
        if f.lower().endswith((".jpg", ".png"))
    ]
    idx = 1
    if existing:
        nums = []
        for f in existing:
            name, _ = os.path.splitext(f)
            # expect something like img_0001
            parts = name.split("_")
            if len(parts) == 2 and parts[1].isdigit():
                nums.append(int(parts[1]))
        if nums:
            idx = max(nums) + 1

    print("Dataset capture started.")
    print("Press 's' to SAVE current frame to data_hotwheels/raw/")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: cannot read frame from camera.")
            continue

        # Optional: show resolution on screen
        h, w = frame.shape[:2]
        cv2.putText(frame, f"{w}x{h}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Capture HotWheels Dataset", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            filename = f"img_{idx:04d}.jpg"
            path = os.path.join(output_dir, filename)
            cv2.imwrite(path, frame)
            print(f"Saved {path}")
            idx += 1

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Capture finished.")

if __name__ == "__main__":
    main()
