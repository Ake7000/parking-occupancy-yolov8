# compute_rois.py
# Uses the orange slot-number detection to automatically build
# the ROIs of the parking slots.

import cv2
import numpy as np
import json
import sys
import os

# IMPORTANT:
# If the modules live in roi_detection/ and this file is imported as
# "roi_detection.compute_rois", the relative import below works.
try:
    from .detect_orange_digits import detect_orange_digits
except ImportError:
    # fallback when run directly: python roi_detection/compute_rois.py
    from detect_orange_digits import detect_orange_digits


# ---------------------------------------------------
# 1. Mask of the WHITE markings
# ---------------------------------------------------
def build_white_mask(frame_bgr):
    """
    Returns a binary mask of the white areas (the markings).
    Uses HSV: white => low saturation, relatively high value.
    Tuned for the mock-up (grey surface, white markings).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Permissive range for "white" (good lighting).
    # any H, low S (<= ~65), medium-to-high V (>= ~145)
    lower_white = np.array([0, 0, 145], dtype=np.uint8)
    upper_white = np.array([179, 65, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


# ---------------------------------------------------
# 2. Find the slot's vertical extent (top/bottom)
# ---------------------------------------------------
def find_vertical_bounds(mask_white, cx, cy):
    """
    For a centre (cx, cy), scans up and down along column x=cx
    until it finds a white line (the horizontal border of the slot).

    Returns (y_top, y_bottom). If none is found, returns fallback values.
    """
    h, w = mask_white.shape
    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))

    y_top = None
    y_bottom = None

    # Search upwards
    for y in range(cy, -1, -1):
        if mask_white[y, cx] > 0:
            y_top = y
            break

    # Search downwards
    for y in range(cy, h):
        if mask_white[y, cx] > 0:
            y_bottom = y
            break

    # Fallback if nothing was found
    if y_top is None:
        y_top = max(cy - 20, 0)
    if y_bottom is None:
        y_bottom = min(cy + 20, h - 1)

    if y_bottom <= y_top:
        y_bottom = min(y_top + 10, h - 1)

    return y_top, y_bottom


# ---------------------------------------------------
# 3. Find the vertical border (left/right)
# ---------------------------------------------------
def find_vertical_border(mask_white, cx, y_top, y_bottom, direction, full_ratio=0.6):
    """
    Searches for the vertical border (white vertical bar) starting at cx,
    between y_top and y_bottom, moving in 'direction':
        direction = -1 => to the left (for slots 1, 2, 3)
        direction = +1 => to the right (for slots 4, 5, 6)

    full_ratio = minimum fraction of the band [y_top, y_bottom] that
                 must be white for a column to count as the border.

    If no column reaches full_ratio, the column with the most
    white pixels is chosen.
    Returns x_border (column index).
    """
    h, w = mask_white.shape
    y_top = int(np.clip(y_top, 0, h - 1))
    y_bottom = int(np.clip(y_bottom, 0, h - 1))
    band_h = max(1, y_bottom - y_top + 1)

    if direction == -1:
        x_range = range(int(cx), -1, -1)
    else:
        x_range = range(int(cx), w)

    best_x = int(cx)
    best_count = -1
    threshold = full_ratio * band_h

    for x in x_range:
        col = mask_white[y_top:y_bottom+1, x]
        count_white = np.count_nonzero(col)

        if count_white > best_count:
            best_count = count_white
            best_x = x

        if count_white >= threshold:
            return x

    return best_x


# ---------------------------------------------------
# 4. Compute the ROIs from the digit centres
# ---------------------------------------------------
def compute_rois_from_centers(frame_bgr, centers, pad_x=5, pad_y=5, buffer_factor=1.15):
    """
    centers: list of 6 centres (cx, cy), in this order:
        1,2,3 left column, top->bottom
        4,5,6 right column, top->bottom   (as returned by detect_orange_digits)

    buffer_factor: >1.0 => enlarges the rectangle around its centre
                   (e.g. 1.15 = +15% in each dimension)

    Returns:
      - list of dicts:
            { "id": slot_id, "x": x, "y": y, "w": w, "h": h }
      - debug image with the rectangles drawn
      - mask of the white markings
    """
    h, w, _ = frame_bgr.shape
    mask_white = build_white_mask(frame_bgr)

    rois = []
    debug_img = frame_bgr.copy()

    for idx, (cx, cy) in enumerate(centers, start=1):
        slot_id = idx

        # 1) find the top and bottom borders
        y_top, y_bottom = find_vertical_bounds(mask_white, cx, cy)

        # small vertical padding
        y0 = max(0, y_top - pad_y)
        y1 = min(h - 1, y_bottom + pad_y)
        height = max(1, y1 - y0)

        # 2) choose the search direction for the vertical border based on the id
        if slot_id in [1, 2, 3]:
            direction = -1   # to the left
        else:
            direction = +1   # to the right

        x_border = find_vertical_border(mask_white, cx, y0, y1, direction)

        half_width = abs(int(cx) - int(x_border))
        if half_width < 5:
            half_width = 5  # minimum fallback

        x0 = int(cx) - half_width
        x1 = int(cx) + half_width

        # small horizontal padding
        x0 = max(0, x0 - pad_x)
        x1 = min(w - 1, x1 + pad_x)
        width = max(1, x1 - x0)

        # ---------- buffer around the centre ----------
        # keep the ROI centre unchanged
        cx_roi = x0 + width / 2.0
        cy_roi = y0 + height / 2.0

        new_w = int(round(width * buffer_factor))
        new_h = int(round(height * buffer_factor))

        new_x0 = int(round(cx_roi - new_w / 2.0))
        new_y0 = int(round(cy_roi - new_h / 2.0))

        # clamp to the image bounds
        new_x0 = max(0, new_x0)
        new_y0 = max(0, new_y0)
        if new_x0 + new_w > w:
            new_w = w - new_x0
        if new_y0 + new_h > h:
            new_h = h - new_y0

        roi = {
            "id": slot_id,
            "x": int(new_x0),
            "y": int(new_y0),
            "w": int(new_w),
            "h": int(new_h)
        }
        rois.append(roi)

        # draw the rectangle on debug_img
        cv2.rectangle(debug_img, (roi["x"], roi["y"]),
                      (roi["x"] + roi["w"], roi["y"] + roi["h"]),
                      (0, 255, 0), 2)
        cv2.putText(debug_img, f"{slot_id}",
                    (roi["x"] + 5, roi["y"] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    rois = sorted(rois, key=lambda r: r["id"])
    return rois, debug_img, mask_white


# ---------------------------------------------------
# 5. Helper used by main_yolo.py
# ---------------------------------------------------
def compute_rois_and_save_from_frame(frame_bgr,
                                     buffer_factor=1.15,
                                     save_path="rois.json"):
    """
    Takes a frame already captured from the camera,
    detects the slot numbers, computes the ROIs and saves them as JSON.

    Returns the list of ROIs (as dicts).
    """
    centers, mask_digits, debug_digits = detect_orange_digits(frame_bgr)

    cv2.imwrite("debug_mask_digits.png", mask_digits)
    cv2.imwrite("debug_digits.png", debug_digits)

    if len(centers) != 6:
        print(f"[Eroare] Ma asteptam la 6 cifre, am gasit: {len(centers)}")
        cv2.imwrite("debug_digits_failed.png", debug_digits)
        print("Am salvat debug_digits_failed.png pentru analiza.")
        return []

    print("Centrele detectate (in ordinea locurilor 1..6):")
    for i, (cx, cy) in enumerate(centers, start=1):
        print(f"  Loc {i}: (x={cx}, y={cy})")

    rois, debug_rois, mask_white = compute_rois_from_centers(
        frame_bgr,
        centers,
        pad_x=5,
        pad_y=5,
        buffer_factor=buffer_factor
    )

    # print to the console
    print("\nROI-uri calculate:")
    for roi in rois:
        print(f"  Loc {roi['id']}: x={roi['x']}, y={roi['y']}, w={roi['w']}, h={roi['h']}")

    # Save the JSON
    with open(save_path, "w") as f:
        json.dump({"rois": rois}, f, indent=4)
    print(f"\nAm salvat roi-urile in {save_path}")

    # Save the debug images
    cv2.imwrite("debug_rois.png", debug_rois)
    cv2.imwrite("debug_white_mask.png", mask_white)
    print("Am salvat debug_rois.png si debug_white_mask.png")

    return rois


# ---------------------------------------------------
# 6. Main - standalone version (optional)
# ---------------------------------------------------
def main():
    """
    Run directly:
      python roi_detection/compute_rois.py             # uses the camera
      python roi_detection/compute_rois.py image.jpg   # uses an image
    """
    # --- 1. Get a frame from an image or the camera ---
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
        cam_index = 0
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

            cv2.imshow("Camera - ROI calibrare", frm)
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

    # --- 2. Compute the ROIs and save them ---
    rois = compute_rois_and_save_from_frame(frame_bgr=frame,
                                            buffer_factor=1.15,
                                            save_path="rois.json")

    # Display for manual inspection
    if rois:
        debug_rois = cv2.imread("debug_rois.png")
        mask_white = cv2.imread("debug_white_mask.png", cv2.IMREAD_GRAYSCALE)

        if debug_rois is not None:
            cv2.imshow("ROI-uri detectate", debug_rois)
        if mask_white is not None:
            cv2.imshow("Masca marcaje albe", mask_white)

        print("Inchide ferestrele cu ESC.")
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
