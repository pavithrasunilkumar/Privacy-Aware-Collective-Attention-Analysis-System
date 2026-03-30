# ============================================================
# privacy.py — Privacy Layer (MediaPipe face detection + blur)
# Uses actual face landmarks so the blur hits the real face,
# not a rough top-35% bbox estimate.
# NO face images are ever stored.
# ============================================================

import cv2
import numpy as np

# ── MediaPipe face detection (accurate region) ───────────────
try:
    import mediapipe as mp
    _mp_fd   = mp.solutions.face_detection
    _face_det = _mp_fd.FaceDetection(model_selection=1,
                                      min_detection_confidence=0.4)
    _USE_MP = True
except Exception:
    _USE_MP = False
    print("[privacy] MediaPipe unavailable — falling back to bbox estimate.")

# ── Blur settings ─────────────────────────────────────────────
BLUR_KERNEL  = (61, 61)   # must be odd; larger = stronger blur
BLUR_SIGMA   = 25
BLUR_PAD     = 10         # extra pixels around detected face box


def blur_faces(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Blur face regions inside each person bounding box.
    detections: list of dicts with keys x1, y1, x2, y2

    Strategy:
      1. Crop the person region.
      2. Run MediaPipe face detection on that crop to get the real face bbox.
      3. Apply Gaussian blur to the face bbox (+ padding) only.
      4. If MediaPipe finds no face, fall back to blurring the upper-30% of
         the person crop (better than nothing, still no face stored).
    """
    if frame is None or frame.size == 0:
        return frame

    h_f, w_f = frame.shape[:2]

    for det in detections:
        x1 = max(0, min(int(det.get("x1", 0)), w_f - 1))
        y1 = max(0, min(int(det.get("y1", 0)), h_f - 1))
        x2 = max(0, min(int(det.get("x2", 0)), w_f - 1))
        y2 = max(0, min(int(det.get("y2", 0)), h_f - 1))

        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        face_found = False

        if _USE_MP:
            rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            results = _face_det.process(rgb)

            if results.detections:
                for face in results.detections:
                    bb  = face.location_data.relative_bounding_box
                    ch, cw = person_crop.shape[:2]

                    # Convert relative coords to pixel coords in the crop
                    fx1 = max(0, int(bb.xmin * cw) - BLUR_PAD)
                    fy1 = max(0, int(bb.ymin * ch) - BLUR_PAD)
                    fx2 = min(cw, int((bb.xmin + bb.width)  * cw) + BLUR_PAD)
                    fy2 = min(ch, int((bb.ymin + bb.height) * ch) + BLUR_PAD)

                    if fx2 - fx1 < 5 or fy2 - fy1 < 5:
                        continue

                    roi = person_crop[fy1:fy2, fx1:fx2]
                    person_crop[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(
                        roi, BLUR_KERNEL, BLUR_SIGMA
                    )
                    face_found = True
                    break   # blur first detected face per person

        # Fallback: blur upper 30% of person bbox if no face found
        if not face_found:
            ch, cw = person_crop.shape[:2]
            fy2 = int(ch * 0.30)
            if fy2 > 4:
                roi = person_crop[0:fy2, 0:cw]
                person_crop[0:fy2, 0:cw] = cv2.GaussianBlur(
                    roi, BLUR_KERNEL, BLUR_SIGMA
                )

        # Write blurred crop back into frame
        frame[y1:y2, x1:x2] = person_crop

    return frame