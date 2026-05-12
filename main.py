# ============================================================
# main.py — ClassWatch Entry Point
# Production-ready: headless-safe, graceful shutdown,
# config from .env, no hardcoded values.
# ============================================================

import cv2
import sys
import signal
import threading
import numpy as np
from datetime import datetime
from collections import defaultdict

# ── Config (reads .env) ───────────────────────────────────────
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    YOLO_MODEL, YOLO_CONF, YOLO_DEVICE, YOLO_MAX_DET,
    SMOOTHING_WINDOW, YAW_THRESHOLD, PITCH_THRESHOLD,
    PRIVACY_ENABLED,
)

# ── Project modules ───────────────────────────────────────────
from logger    import AttentionLogger
from dashboard import (start_web_dashboard, update_student,
                        print_live_dashboard, run_final_dashboard,
                        push_frame, register_shutdown)
from privacy   import blur_faces
from utils     import FPSCounter, draw_person_box, draw_hud, ensure_dirs

# ── YOLO + MediaPipe ──────────────────────────────────────────
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

_model = YOLO(YOLO_MODEL)
_model.to(YOLO_DEVICE)

_mp_fd     = mp.solutions.face_detection
_mp_fm     = mp.solutions.face_mesh
_face_det  = _mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.3)
_face_mesh = _mp_fm.FaceMesh(refine_landmarks=True, max_num_faces=1)

# ── Logger ────────────────────────────────────────────────────
_logger = AttentionLogger()

# ── Smoothing ─────────────────────────────────────────────────
def _smooth(history: list, window: int) -> str:
    recent = history[-window:]
    return max(set(recent), key=recent.count)

# ── Head-pose attention ───────────────────────────────────────
def get_attention_state(frame, box):
    x1, y1, x2, y2 = box
    pad = 15
    h_f, w_f = frame.shape[:2]
    cx1 = max(0, x1-pad); cy1 = max(0, y1-pad)
    cx2 = min(w_f, x2+pad); cy2 = min(h_f, y2+pad)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return "Distracted"
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    fd  = _face_det.process(rgb)
    if not fd.detections:
        return "Distracted"
    det = fd.detections[0]
    bb  = det.location_data.relative_bounding_box
    ch, cw = crop.shape[:2]
    fx1 = max(0, int(bb.xmin * cw))
    fy1 = max(0, int(bb.ymin * ch))
    fx2 = min(cw, int((bb.xmin + bb.width) * cw))
    fy2 = min(ch, int((bb.ymin + bb.height) * ch))
    face = crop[fy1:fy2, fx1:fx2]
    if face.size == 0:
        return "Distracted"
    mr = _face_mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    if not mr.multi_face_landmarks:
        return "Distracted"
    lms = mr.multi_face_landmarks[0]
    fh, fw = face.shape[:2]
    p2d, p3d = [], []
    for idx, lm in enumerate(lms.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
            px, py = int(lm.x*fw), int(lm.y*fh)
            p2d.append([px, py]); p3d.append([px, py, lm.z])
    if len(p2d) < 6:
        return "Distracted"
    p2d = np.array(p2d, dtype=np.float64)
    p3d = np.array(p3d, dtype=np.float64)
    cam  = np.array([[fw,0,fw/2],[0,fw,fh/2],[0,0,1]], dtype=np.float64)
    dist = np.zeros((4,1), dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(p3d, p2d, cam, dist)
    if not ok:
        return "Distracted"
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch = angles[0] * 360
    yaw   = angles[1] * 360
    if abs(yaw) < YAW_THRESHOLD and abs(pitch) < PITCH_THRESHOLD:
        return "Attentive"
    return "Distracted"

# ── Graceful shutdown flag ────────────────────────────────────
_stop_event = threading.Event()

def _signal_handler(sig, frame):
    print("\n[main] Signal received — shutting down…")
    _stop_event.set()

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ── Main loop ─────────────────────────────────────────────────
def main():
    ensure_dirs("data", "outputs")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[main] ERROR: Cannot open camera index {CAMERA_INDEX}.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    fps_counter    = FPSCounter(window=30)
    session_start  = datetime.now().strftime("%H:%M:%S")
    student_history: dict = {}
    _sum_pct  = 0.0
    _frame_idx = 0

    # Register shutdown callback so browser /shutdown route works too
    register_shutdown(_stop_event)

    # Start web dashboard (opens browser automatically)
    start_web_dashboard()

    print("\n  ┌──────────────────────────────────────────────┐")
    print("  │  ClassWatch — Attention Analysis System       │")
    print("  │  Dashboard  →  http://localhost:5000          │")
    print("  │  Press  q   or Ctrl-C to stop                │")
    print("  └──────────────────────────────────────────────┘\n")

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("\n[main] Camera read failed — stopping.")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        _frame_idx += 1

        # ── Detect + Track ────────────────────────────────
        results = _model.track(
            frame, persist=True,
            tracker="bytetrack.yaml",
            conf=YOLO_CONF,
            max_det=YOLO_MAX_DET,
            verbose=False
        )

        attentive = 0; total = 0
        blur_dets: list[dict] = []
        current_states: dict = {}

        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                if int(box.cls[0]) != 0 or box.id is None: continue
                tid = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                total += 1
                blur_dets.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2})

                raw = get_attention_state(frame, (x1, y1, x2, y2))
                student_history.setdefault(tid, []).append(raw)
                smoothed = _smooth(student_history[tid], SMOOTHING_WINDOW)
                current_states[tid] = smoothed

                update_student(tid, smoothed)

                if smoothed == "Attentive":
                    attentive += 1
                draw_person_box(frame, x1, y1, x2, y2, tid, smoothed)

        # ── Privacy ───────────────────────────────────────
        if PRIVACY_ENABLED and blur_dets:
            blur_faces(frame, blur_dets)

        # ── HUD ───────────────────────────────────────────
        fps = fps_counter.tick()
        draw_hud(frame, attentive, total, fps, PRIVACY_ENABLED)

        # ── Log ───────────────────────────────────────────
        _logger.log_attention({
            "attentive_count": attentive,
            "total_students":  total,
        })

        # ── Dashboard push ────────────────────────────────
        _sum_pct += (attentive / total * 100) if total else 0.0
        avg_pct   = round(_sum_pct / _frame_idx, 1)
        print_live_dashboard(attentive, total, avg_pct, session_start, fps)

        # ── Stream frame to browser ───────────────────────
        push_frame(frame)

        # ── Headless-safe quit check ──────────────────────
        # waitKey(1) works fine with no window — returns -1 silently
        if cv2.waitKey(1) & 0xFF == ord("q"):
            _stop_event.set()

    # ── Cleanup ───────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("\n\n  Stopping capture…\n")
    run_final_dashboard()


if __name__ == "__main__":
    main()