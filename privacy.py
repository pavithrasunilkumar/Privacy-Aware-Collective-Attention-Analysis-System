# ============================================================
# privacy.py — Privacy Layer (MediaPipe face detection + blur)
# Uses actual face landmarks so the blur hits the real face,
# not a rough top-35% bbox estimate.
# NO face images are ever stored.
# ============================================================

import cv2
import numpy as np
import contextlib
import io

# ── MediaPipe face detection (accurate region) ───────────────
_USE_MP = False
_face_det = None
_mp_checked = False


def _ensure_mediapipe():
    global _USE_MP, _face_det, _mp_checked

    if _mp_checked:
        return

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import mediapipe as mp

        _mp_fd = mp.solutions.face_detection
        _face_det = _mp_fd.FaceDetection(model_selection=1,
                                          min_detection_confidence=0.4)
        _USE_MP = True
    except Exception:
        _USE_MP = False
        _face_det = None
        print("[privacy] MediaPipe unavailable — falling back to bbox estimate.")
    finally:
        _mp_checked = True

# ── Blur settings ─────────────────────────────────────────────
BLUR_KERNEL  = (61, 61)   # must be odd; larger = stronger blur
BLUR_SIGMA   = 25
BLUR_PAD     = 10         # extra pixels around detected face box
SMOOTHING_ALPHA = 0.8      # higher = more stable blur, lower = more responsive

_face_cache = {}


def _cache_key(det: dict) -> str:
    track_id = det.get("track_id")
    if track_id is not None:
        return str(track_id)
    return f"{int(det.get('x1', 0))}:{int(det.get('y1', 0))}:{int(det.get('x2', 0))}:{int(det.get('y2', 0))}"


def _smooth_box(previous_box, current_box):
    if previous_box is None:
        return current_box

    return tuple(
        previous_box[index] * SMOOTHING_ALPHA + current_box[index] * (1.0 - SMOOTHING_ALPHA)
        for index in range(4)
    )


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

    _ensure_mediapipe()

    h_f, w_f = frame.shape[:2]

    for det in detections:
        cache_key = _cache_key(det)
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
        cached_rel_box = _face_cache.get(cache_key)

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

                    current_rel_box = (
                        fx1 / cw,
                        fy1 / ch,
                        fx2 / cw,
                        fy2 / ch,
                    )
                    smoothed_rel_box = _smooth_box(cached_rel_box, current_rel_box)
                    _face_cache[cache_key] = smoothed_rel_box

                    sfx1 = max(0, min(int(smoothed_rel_box[0] * cw), cw - 1))
                    sfy1 = max(0, min(int(smoothed_rel_box[1] * ch), ch - 1))
                    sfx2 = max(0, min(int(smoothed_rel_box[2] * cw), cw))
                    sfy2 = max(0, min(int(smoothed_rel_box[3] * ch), ch))

                    if sfx2 - sfx1 < 5 or sfy2 - sfy1 < 5:
                        continue

                    roi = person_crop[sfy1:sfy2, sfx1:sfx2]
                    person_crop[sfy1:sfy2, sfx1:sfx2] = cv2.GaussianBlur(
                        roi, BLUR_KERNEL, BLUR_SIGMA
                    )
                    face_found = True
                    break   # blur first detected face per person

        # Fallback: blur upper 30% of person bbox if no face found
        if not face_found:
            if cached_rel_box is not None:
                ch, cw = person_crop.shape[:2]
                fx1 = max(0, min(int(cached_rel_box[0] * cw), cw - 1))
                fy1 = max(0, min(int(cached_rel_box[1] * ch), ch - 1))
                fx2 = max(0, min(int(cached_rel_box[2] * cw), cw))
                fy2 = max(0, min(int(cached_rel_box[3] * ch), ch))

                if fx2 - fx1 >= 5 and fy2 - fy1 >= 5:
                    roi = person_crop[fy1:fy2, fx1:fx2]
                    person_crop[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(
                        roi, BLUR_KERNEL, BLUR_SIGMA
                    )
                    face_found = True

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