import cv2
import contextlib
import io
import sys
from datetime import datetime

# ── Project modules ───────────────────────────────────────────────────────────
from logger    import AttentionLogger
from utils     import (FPSCounter, draw_person_box, draw_hud, ensure_dirs)

# Instantiate logger (handles 2-second time-gating internally)
_logger = AttentionLogger()

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX      = 0
FRAME_WIDTH       = 1280
FRAME_HEIGHT      = 720
YOLO_MODEL        = "yolov8n.pt"
YOLO_CONF         = 0.6
YOLO_DEVICE       = "cuda"
SMOOTHING_WINDOW  = 11
PRIVACY_ENABLED   = True
WINDOW_TITLE      = "ClassWatch — Privacy-Aware Attention System"

YAW_THRESHOLD   = 55
PITCH_THRESHOLD = 50

# ── YOLO + MediaPipe setup ───────────────────────────────────────────────────
import numpy as np

_model      = None
_face_det   = None
_face_mesh  = None
_face_models_checked = False
_face_models_ready = False
_tracking_checked = False
_tracking_ready = False
_dashboard_loaded = False
_cuda_checked = False


def _ensure_dashboard_api():
    global _dashboard_loaded, start_web_dashboard, update_student
    global print_live_dashboard, run_final_dashboard

    if _dashboard_loaded:
        return

    from dashboard import (start_web_dashboard, update_student,
                           print_live_dashboard, run_final_dashboard)

    _dashboard_loaded = True


def _ensure_vision_models():
    global _model, _face_det, _face_mesh, YOLO_DEVICE, _cuda_checked
    global _face_models_checked, _face_models_ready

    if _model is None:
        from ultralytics import YOLO
        import torch

        if not _cuda_checked:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is required but not available. Install a CUDA-enabled PyTorch build and run on the RTX 4060."
                )
            _cuda_checked = True

        _model = YOLO(YOLO_MODEL)
        _model.to(YOLO_DEVICE)

    if _face_models_checked:
        return

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import mediapipe as mp

        _mp_fd = mp.solutions.face_detection
        _mp_fm = mp.solutions.face_mesh
        _face_det = _mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.3)
        _face_mesh = _mp_fm.FaceMesh(refine_landmarks=True, max_num_faces=1)
        _face_models_ready = True
    except Exception:
        _face_det = None
        _face_mesh = None
        _face_models_ready = False
        print("[main] MediaPipe unavailable — face attention will default to Distracted.")
    finally:
        _face_models_checked = True

# Track IDs assigned by ByteTrack across frames
_track_id_map = {}


def detect_persons(frame):
    _ensure_vision_models()

    global _tracking_checked, _tracking_ready

    if _tracking_ready or not _tracking_checked:
        results = _model.track(
            frame, persist=True,
            tracker="bytetrack.yaml",
            conf=YOLO_CONF,
            verbose=False,
        )
        _tracking_checked = True
        _tracking_ready = True

        out = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if int(box.cls[0]) != 0:
                    continue
                if box.id is None:
                    continue
                tid = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                out.append((x1, y1, x2, y2, tid))
        return out

    raise RuntimeError("Unexpected tracker state: tracking should be initialized on first use.")


def update_tracker(frame, detections):
    return detections


def get_attention_state(frame, box):
    _ensure_vision_models()

    if not _face_models_ready:
        return "Distracted"

    x1, y1, x2, y2 = box
    pad = 15
    h_f, w_f = frame.shape[:2]
    cx1 = max(0, x1 - pad);  cy1 = max(0, y1 - pad)
    cx2 = min(w_f, x2 + pad); cy2 = min(h_f, y2 + pad)
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
    fx2 = min(cw, int((bb.xmin + bb.width)  * cw))
    fy2 = min(ch, int((bb.ymin + bb.height) * ch))
    face = crop[fy1:fy2, fx1:fx2]
    if face.size == 0:
        return "Distracted"

    mesh = _face_mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    if not mesh.multi_face_landmarks:
        return "Distracted"

    lms = mesh.multi_face_landmarks[0]
    fh, fw = face.shape[:2]
    p2d, p3d = [], []
    for idx, lm in enumerate(lms.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
            px, py = int(lm.x * fw), int(lm.y * fh)
            p2d.append([px, py])
            p3d.append([px, py, lm.z])
    if len(p2d) < 6:
        return "Distracted"

    p2d = np.array(p2d, dtype=np.float64)
    p3d = np.array(p3d, dtype=np.float64)
    cam  = np.array([[fw, 0, fw/2], [0, fw, fh/2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
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


def _smooth(history: list, window: int) -> str:
    recent = history[-window:]
    return max(set(recent), key=recent.count)


def main() -> None:
    ensure_dirs("data", "outputs")
    _logger.reset_session_logs()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[main] ERROR: Cannot open camera index {CAMERA_INDEX}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    fps_counter    = FPSCounter(window=30)
    session_start  = datetime.now().strftime("%H:%M:%S")
    student_history: dict = {}

    _ensure_dashboard_api()
    from privacy import blur_faces

    start_web_dashboard()

    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  ClassWatch — Attention Analysis System  │")
    print("  │  Dashboard → http://localhost:5000        │")
    print("  │  Press  q  to stop and generate reports  │")
    print("  └─────────────────────────────────────────┘\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n[main] Camera read failed — stopping.")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        detections = detect_persons(frame)
        tracked = update_tracker(frame, detections)

        current_states = {}
        for (x1, y1, x2, y2, tid) in tracked:
            state = get_attention_state(frame, (x1, y1, x2, y2))
            student_history.setdefault(tid, []).append(state)
            smoothed = _smooth(student_history[tid], SMOOTHING_WINDOW)
            current_states[tid] = smoothed
            update_student(tid, smoothed)

        if PRIVACY_ENABLED and tracked:
            blur_dets = [{"x1":x1,"y1":y1,"x2":x2,"y2":y2,"track_id":tid}
                         for (x1,y1,x2,y2,tid) in tracked]
            blur_faces(frame, blur_dets)

        for (x1, y1, x2, y2, tid) in tracked:
            draw_person_box(frame, x1, y1, x2, y2, tid, current_states[tid])

        attentive  = sum(1 for s in current_states.values() if s == "Attentive")
        total      = len(current_states)
        distracted = total - attentive

        _logger.log_attention({
            "attentive_count": attentive,
            "total_students":  total,
        })

        fps = fps_counter.tick()
        draw_hud(frame, attentive, total, fps, PRIVACY_ENABLED)

        avg_pct = round(attentive / total * 100, 1) if total else 0.0
        print_live_dashboard(attentive, total, avg_pct, session_start, fps)

        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n\n  Stopping capture…\n")

    run_final_dashboard()


if __name__ == "__main__":
    main()