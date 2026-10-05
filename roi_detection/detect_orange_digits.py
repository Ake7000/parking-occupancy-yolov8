# detect_orange_digits.py
# Detects the red / orange slot numbers of the parking lot.
# Debug: saves the mask and the frame with the centres marked.

import cv2
import numpy as np
import sys
import os

# ------------ COLOUR SETTINGS (HSV) ------------
# H ∈ [0,179], S,V ∈ [0,255]
# Range for red/orange (two bands, because red wraps around the hue scale)
LOWER_RED1 = np.array([0, 90, 70])
UPPER_RED1 = np.array([12, 255, 255])
LOWER_RED2 = np.array([168, 90, 70])
UPPER_RED2 = np.array([179, 255, 255])


def detect_orange_digits(frame_bgr, min_area=100, max_components=6):
    """
    Returns:
      centers_sorted: list of (x,y) digit centres, IN THIS ORDER:
          index 0 -> slot 1 (left top)
          index 1 -> slot 2 (left middle)
          index 2 -> slot 3 (left bottom)
          index 3 -> slot 4 (right top)
          index 4 -> slot 5 (right middle)
          index 5 -> slot 6 (right bottom)
      mask: binary mask of the red areas,
      debug_img: frame with the digits and labels 1..6 drawn.

    Logic:
      - detect all red contours
      - keep only the largest `max_components`
      - split them into 2 columns (left/right) by x
      - sort each column by y (top->bottom)
      - map to labels 1..6
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Mask for the two red ranges
    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Compute each contour's area and keep only the large ones
    components = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        components.append((area, cnt))

    # Sort by area (descending) and keep only the first 6
    components.sort(key=lambda x: x[0], reverse=True)
    components = components[:max_components]

    centers = []
    for area, cnt in components:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    debug_img = frame_bgr.copy()

    labels_and_centers = []

    if len(centers) == 6:
        # 1) sort by x to split into left/right
        centers_by_x = sorted(centers, key=lambda p: p[0])
        left = centers_by_x[:3]
        right = centers_by_x[3:]

        # 2) sort each group by y (top -> bottom)
        left_sorted = sorted(left, key=lambda p: p[1])
        right_sorted = sorted(right, key=lambda p: p[1])

        # 3) map to labels
        # left: top=1, middle=2, bottom=3
        labels_and_centers.append((1, left_sorted[0]))
        labels_and_centers.append((2, left_sorted[1]))
        labels_and_centers.append((3, left_sorted[2]))
        # right: top=4, middle=5, bottom=6
        labels_and_centers.append((4, right_sorted[0]))
        labels_and_centers.append((5, right_sorted[1]))
        labels_and_centers.append((6, right_sorted[2]))

        # sort by label to get [1..6] in order
        labels_and_centers.sort(key=lambda t: t[0])

    else:
        # fallback: if there are not exactly 6, use a simple sort (top-bottom, left-right)
        simple_sorted = sorted(centers, key=lambda p: (p[1], p[0]))
        for idx, c in enumerate(simple_sorted, start=1):
            labels_and_centers.append((idx, c))

    # Build the list of centres in label order
    centers_sorted = [c for (label, c) in labels_and_centers]

    # Draw circles + labels (1..6)
    for label, (cx, cy) in labels_and_centers:
        cv2.circle(debug_img, (cx, cy), 6, (0, 255, 255), -1)
        cv2.putText(debug_img, str(label), (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return centers_sorted, mask, debug_img


def main():
    """
    Usage:
      python detect_orange_digits.py            # uses the camera
      python detect_orange_digits.py image.jpg  # uses an image
    """
    # --- 1. Frame from an image or the camera ---
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print(f"Eroare: fisierul {img_path} nu exista.")
            return
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Eroare: nu pot citi imaginea {img_path}.")
            return
        print(f"Folosesc imaginea: {img_path}")
    else:
        cam_index = 0  # camera index
        cap = cv2.VideoCapture(cam_index)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1152)

        if not cap.isOpened():
            print(f"Eroare: nu pot deschide camera cu index {cam_index}")
            return

        print("Apasa SPACE pentru a captura un frame, 'q' pentru iesire.")
        frame = None

        while True:
            ret, frm = cap.read()
            if not ret:
                print("Nu pot citi frame de la camera.")
                continue

            cv2.imshow("Camera - detectare cifre", frm)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                frame = frm.copy()
                print("Cadru capturat.")
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

    # --- 2. Detection ---
    centers, mask, debug_img = detect_orange_digits(frame)

    print(f"Numar de regiuni portocalii detectate (dupa filtrare): {len(centers)}")
    for i, (cx, cy) in enumerate(centers, start=1):
        print(f"  Loc {i}: (x={cx}, y={cy})")

    cv2.imwrite("debug_mask_digits.png", mask)
    cv2.imwrite("debug_digits.png", debug_img)
    print("Am salvat debug_mask_digits.png si debug_digits.png")

    cv2.imshow("Masca cifrelor (debug)", mask)
    cv2.imshow("Cifre detectate (debug)", debug_img)
    print("Inchide ferestrele cu ESC.")
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # ESC
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
