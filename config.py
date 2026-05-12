# ============================================================
# config.py — Central configuration loader
# Reads from .env file (or environment variables directly)
# Import this in main.py and dashboard.py instead of
# hardcoding values.
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env if present; silently skips if missing

def _bool(key, default=True):
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")

def _int(key, default):
    try:    return int(os.getenv(key, default))
    except: return default

def _float(key, default):
    try:    return float(os.getenv(key, default))
    except: return default

# ── Server ───────────────────────────────────────────────────
WEB_PORT    = _int("WEB_PORT", 5000)
WEB_HOST    = os.getenv("WEB_HOST", "0.0.0.0")
SECRET_KEY  = os.getenv("SECRET_KEY", "dev-secret-change-me")

# ── Stream auth ──────────────────────────────────────────────
STREAM_PASSWORD = os.getenv("STREAM_PASSWORD", "")   # empty = no auth

# ── Camera ───────────────────────────────────────────────────
CAMERA_INDEX  = _int("CAMERA_INDEX", 0)
FRAME_WIDTH   = _int("FRAME_WIDTH", 1280)
FRAME_HEIGHT  = _int("FRAME_HEIGHT", 720)

# ── YOLO ─────────────────────────────────────────────────────
YOLO_MODEL   = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_CONF    = _float("YOLO_CONF", 0.6)
YOLO_DEVICE  = os.getenv("YOLO_DEVICE", "cuda")
YOLO_MAX_DET = _int("YOLO_MAX_DET", 30)

# ── Attention ─────────────────────────────────────────────────
SMOOTHING_WINDOW      = _int("SMOOTHING_WINDOW", 11)
YAW_THRESHOLD         = _int("YAW_THRESHOLD", 55)
PITCH_THRESHOLD       = _int("PITCH_THRESHOLD", 50)
DISTRACTION_THRESHOLD = _float("DISTRACTION_THRESHOLD", 50.0)

# ── Privacy ───────────────────────────────────────────────────
PRIVACY_ENABLED = _bool("PRIVACY_ENABLED", True)

# ── Logging ───────────────────────────────────────────────────
LOG_INTERVAL = _int("LOG_INTERVAL", 2)

# ── Paths ─────────────────────────────────────────────────────
LOG_PATH         = "data/attention_log.csv"
STUDENT_LOG_PATH = "data/student_log.csv"
GRAPH_PATH       = "outputs/attention_graph.png"
SUMMARY_PATH     = "outputs/summary.txt"