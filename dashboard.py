# ============================================================
# dashboard.py — ClassWatch Web Dashboard
# Beautiful production-ready UI with:
#   • MJPEG camera stream with toggle on/off
#   • Sidebar navigation
#   • Animated KPI cards, gauge, live charts
#   • Distraction log, per-student cards with sparklines
#   • Session summary banner on Q press
#   • Fully responsive dark theme
# ============================================================

import threading
import webbrowser
import queue
import cv2
from datetime import datetime
from flask import Flask, render_template_string, jsonify, Response
from flask_socketio import SocketIO
from analytics import (compute_statistics, compute_student_scores,
                        generate_graph, generate_summary)
from utils import print_section
from config import (
    WEB_PORT, WEB_HOST, SECRET_KEY, STREAM_PASSWORD,
    LOG_PATH, STUDENT_LOG_PATH, GRAPH_PATH, SUMMARY_PATH,
    DISTRACTION_THRESHOLD,
)

app      = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── MJPEG frame buffer ────────────────────────────────────────
_frame_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=1)
_last_jpeg:   bytes                = b""
_frame_lock   = threading.Lock()
_cam_enabled  = True   # toggled via /toggle_camera

# ── attention state ───────────────────────────────────────────
_state = {
    "pct": 0.0, "avg_pct": 0.0,
    "attentive": 0, "total": 0,
    "fps": 0.0, "start_time": "—",
    "students":        {},
    "timeline":        [],
    "distraction_log": [],
    "peak_pct": 0.0,   "peak_time": "—",
    "low_pct":  100.0, "low_time":  "—",
}
_last_was_distracted = False
_state_lock    = threading.Lock()
_final_summary = None

# ─────────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ClassWatch</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet"/>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
/* ── TOKENS ── */
:root{
  --bg:        #060810;
  --bg2:       #0b0f1a;
  --surface:   #0f1421;
  --surface2:  #151d2e;
  --surface3:  #1c2638;
  --border:    rgba(255,255,255,0.055);
  --border2:   rgba(255,255,255,0.1);
  --accent:    #00f5a8;
  --accent-d:  #00c485;
  --blue:      #4d9eff;
  --red:       #ff4d6d;
  --amber:     #ffb347;
  --purple:    #a78bfa;
  --text:      #e2e8f4;
  --text2:     #8b97b0;
  --text3:     #4a5568;
  --r:         'Inter',sans-serif;
  --m:         'JetBrains Mono',monospace;
  --ease:      cubic-bezier(.4,0,.2,1);
  --glow:      0 0 40px rgba(0,245,168,.08);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--text);font-family:var(--r);font-size:14px;-webkit-font-smoothing:antialiased}

/* ── APP SHELL ── */
.app{display:grid;grid-template-columns:64px 1fr;grid-template-rows:1fr;height:100vh}

/* ── SIDEBAR ── */
.sidebar{
  background:var(--surface);
  border-right:1px solid var(--border);
  display:flex;flex-direction:column;align-items:center;
  padding:16px 0;gap:4px;z-index:50;
}
.sidebar-logo{
  width:40px;height:40px;border-radius:12px;margin-bottom:20px;
  background:linear-gradient(135deg,var(--accent),var(--blue));
  display:grid;place-items:center;flex-shrink:0;cursor:pointer;
  box-shadow:0 4px 20px rgba(0,245,168,.25);
}
.sidebar-logo svg{width:20px;height:20px}
.nav-btn{
  width:40px;height:40px;border-radius:10px;border:none;background:none;
  color:var(--text3);cursor:pointer;display:grid;place-items:center;
  transition:all .2s var(--ease);position:relative;
}
.nav-btn:hover{background:var(--surface2);color:var(--text2)}
.nav-btn.active{background:rgba(0,245,168,.1);color:var(--accent)}
.nav-btn.active::before{
  content:'';position:absolute;left:0;top:50%;transform:translateY(-50%);
  width:3px;height:22px;background:var(--accent);border-radius:0 3px 3px 0;
}
.nav-btn svg{width:18px;height:18px;stroke-width:1.8}
.sidebar-spacer{flex:1}
.sidebar-bottom{display:flex;flex-direction:column;align-items:center;gap:4px}

/* ── MAIN ── */
.main{display:flex;flex-direction:column;overflow:hidden;background:var(--bg)}

/* ── TOPBAR ── */
.topbar{
  height:56px;flex-shrink:0;
  background:var(--surface);border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  padding:0 20px;gap:12px;
}
.topbar-left{display:flex;align-items:center;gap:12px}
.page-title{font-size:15px;font-weight:600;color:var(--text)}
.breadcrumb{font-size:12px;color:var(--text3);font-family:var(--m)}
.topbar-right{display:flex;align-items:center;gap:10px}

/* ── BADGES ── */
.badge{
  display:inline-flex;align-items:center;gap:5px;
  border-radius:20px;padding:4px 10px;
  font-family:var(--m);font-size:10px;font-weight:500;letter-spacing:.5px;
}
.badge-live{background:rgba(0,245,168,.1);border:1px solid rgba(0,245,168,.2);color:var(--accent)}
.badge-ended{background:rgba(255,179,71,.1);border:1px solid rgba(255,179,71,.2);color:var(--amber)}
.badge-off{background:rgba(255,77,109,.1);border:1px solid rgba(255,77,109,.2);color:var(--red)}
.pulse{width:5px;height:5px;border-radius:50%;background:currentColor;animation:pulse 1.5s ease infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.6)}}

/* ── BUTTONS ── */
.btn{
  display:inline-flex;align-items:center;gap:6px;
  border-radius:8px;border:1px solid var(--border2);
  background:var(--surface2);color:var(--text2);
  font-family:var(--r);font-size:12px;font-weight:500;
  padding:6px 12px;cursor:pointer;transition:all .18s var(--ease);
}
.btn:hover{background:var(--surface3);color:var(--text);border-color:var(--border2)}
.btn svg{width:14px;height:14px;stroke-width:2}
.btn-accent{background:rgba(0,245,168,.1);border-color:rgba(0,245,168,.25);color:var(--accent)}
.btn-accent:hover{background:rgba(0,245,168,.2)}
.btn-red{background:rgba(255,77,109,.08);border-color:rgba(255,77,109,.2);color:var(--red)}
.btn-red:hover{background:rgba(255,77,109,.18)}
.btn-sm{padding:5px 10px;font-size:11px;border-radius:6px}

/* ── TOGGLE ── */
.toggle-wrap{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--text2)}
.toggle{position:relative;width:36px;height:20px;cursor:pointer}
.toggle input{opacity:0;width:0;height:0}
.toggle-slider{
  position:absolute;inset:0;border-radius:20px;
  background:var(--surface3);border:1px solid var(--border2);
  transition:all .25s var(--ease);
}
.toggle-slider::before{
  content:'';position:absolute;left:3px;top:50%;transform:translateY(-50%);
  width:14px;height:14px;border-radius:50%;
  background:var(--text3);transition:all .25s var(--ease);
}
.toggle input:checked + .toggle-slider{background:rgba(0,245,168,.2);border-color:rgba(0,245,168,.4)}
.toggle input:checked + .toggle-slider::before{background:var(--accent);transform:translate(16px,-50%)}

/* ── SCROLLABLE CONTENT ── */
.content{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}
.content::-webkit-scrollbar{width:4px}
.content::-webkit-scrollbar-thumb{background:var(--surface3);border-radius:4px}

/* ── PAGE (tabs) ── */
.page{display:none;flex-direction:column;gap:16px;flex:1}
.page.active{display:flex}

/* ── GRID LAYOUTS ── */
.grid-cols-3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.grid-cols-2{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
.grid-cam-kpi{display:grid;grid-template-columns:1fr 340px;gap:14px}
.grid-kpi-4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}

/* ── CARD ── */
.card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:14px;padding:18px;
  transition:border-color .2s var(--ease);
}
.card:hover{border-color:var(--border2)}
.card-sm{padding:14px 16px;border-radius:12px}
.card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
.card-label{
  font-size:10px;font-weight:600;text-transform:uppercase;
  letter-spacing:1.2px;color:var(--text3);
  display:flex;align-items:center;gap:6px;
}
.label-dot{width:6px;height:6px;border-radius:50%;background:var(--accent);flex-shrink:0}
.label-dot.blue{background:var(--blue)}
.label-dot.red{background:var(--red)}
.label-dot.amber{background:var(--amber)}
.label-dot.purple{background:var(--purple)}

/* ── KPI CARD ── */
.kpi-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:14px;padding:16px 18px;position:relative;overflow:hidden;
  transition:all .2s var(--ease);
}
.kpi-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--kc,var(--accent));
}
.kpi-card:hover{border-color:var(--border2);transform:translateY(-1px)}
.kpi-label{font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:1px;color:var(--text3);margin-bottom:10px}
.kpi-value{font-family:var(--m);font-size:28px;font-weight:600;line-height:1;color:var(--text);margin-bottom:5px;transition:color .3s}
.kpi-sub{font-size:11px;color:var(--text3)}
.kpi-trend{display:flex;align-items:center;gap:4px;font-size:10px;margin-top:8px;font-family:var(--m)}

/* ── GAUGE ── */
.gauge-card{display:flex;flex-direction:column;align-items:center;padding-top:20px}
.gauge-wrap{position:relative;width:148px;height:84px;margin-bottom:6px}
.gauge-svg{width:148px;height:84px;overflow:visible}
.g-track{fill:none;stroke:var(--surface3);stroke-width:13;stroke-linecap:round}
.g-fill{fill:none;stroke-width:13;stroke-linecap:round;stroke-dasharray:231;stroke-dashoffset:231;transition:stroke-dashoffset .7s var(--ease),stroke .4s}
.gauge-num{
  position:absolute;bottom:0;left:50%;transform:translateX(-50%);
  font-family:var(--m);font-size:24px;font-weight:600;
  color:var(--text);white-space:nowrap;text-align:center;
}
.gauge-sub{font-size:10px;color:var(--text3);text-align:center;margin-top:2px}
.gauge-splits{display:flex;gap:16px;margin-top:12px}
.gauge-split{text-align:center}
.gauge-split-val{font-family:var(--m);font-size:16px;font-weight:600}
.gauge-split-label{font-size:9px;color:var(--text3);margin-top:2px;text-transform:uppercase;letter-spacing:.8px}

/* ── CAMERA ── */
.camera-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:14px;overflow:hidden;display:flex;flex-direction:column;
}
.camera-topbar{
  padding:12px 16px;display:flex;align-items:center;
  justify-content:space-between;border-bottom:1px solid var(--border);
  background:var(--surface2);
}
.camera-label{font-size:11px;font-weight:600;color:var(--text2);display:flex;align-items:center;gap:7px;text-transform:uppercase;letter-spacing:.8px}
.rec-dot{width:7px;height:7px;border-radius:50%;background:var(--red);animation:pulse 1.4s ease infinite}
.camera-controls{display:flex;align-items:center;gap:8px}
.camera-body{position:relative;background:#000;flex:1;display:flex;align-items:center;justify-content:center;min-height:320px}
#videoFeed{width:100%;height:100%;object-fit:contain;display:block;transition:opacity .4s}
#videoFeed.hidden{opacity:0}
.camera-off-overlay{
  position:absolute;inset:0;display:none;
  flex-direction:column;align-items:center;justify-content:center;gap:12px;
  background:var(--surface);
}
.camera-off-overlay.show{display:flex}
.camera-off-icon{width:48px;height:48px;border-radius:50%;background:var(--surface3);display:grid;place-items:center;color:var(--text3)}
.camera-off-icon svg{width:22px;height:22px}
.camera-footer{
  padding:10px 16px;background:var(--surface2);border-top:1px solid var(--border);
  display:flex;gap:16px;font-family:var(--m);font-size:10px;color:var(--text3);
}
.cam-stat{display:flex;align-items:center;gap:5px}
.cam-stat-val{color:var(--text);font-weight:500}
.cam-stat-badge{
  padding:1px 6px;border-radius:4px;font-size:9px;font-weight:600;
  background:rgba(0,245,168,.1);color:var(--accent);letter-spacing:.3px;
}

/* ── CHARTS ── */
canvas{display:block}

/* ── DISTRACTION LOG ── */
.dist-list{display:flex;flex-direction:column;gap:6px;max-height:280px;overflow-y:auto}
.dist-list::-webkit-scrollbar{width:3px}
.dist-list::-webkit-scrollbar-thumb{background:var(--surface3);border-radius:2px}
.dist-row{
  display:flex;align-items:center;gap:10px;padding:8px 12px;
  background:var(--surface2);border-radius:8px;
  border-left:2px solid var(--red);
  transition:background .15s;
}
.dist-row:hover{background:var(--surface3)}
.dist-time{font-family:var(--m);font-size:10px;color:var(--amber);width:66px;flex-shrink:0}
.dist-pct{font-family:var(--m);font-size:12px;color:var(--red);font-weight:600;width:36px;flex-shrink:0}
.dist-bar-bg{flex:1;height:3px;background:var(--surface3);border-radius:2px}
.dist-bar-fg{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--red),var(--amber))}
.dist-empty{color:var(--text3);font-size:13px;padding:12px 0;display:flex;align-items:center;gap:8px}
.dist-empty-icon{width:28px;height:28px;border-radius:8px;background:rgba(0,245,168,.08);display:grid;place-items:center;color:var(--accent)}

/* ── STUDENT CARDS ── */
.students-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.sc{
  background:var(--surface2);border:1px solid var(--border);
  border-radius:12px;padding:14px;
  transition:all .2s var(--ease);position:relative;overflow:hidden;
}
.sc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--sc-c,var(--surface3))}
.sc.att{border-color:rgba(0,245,168,.2);--sc-c:var(--accent)}
.sc.dist{border-color:rgba(255,77,109,.2);--sc-c:var(--red)}
.sc:hover{transform:translateY(-2px);border-color:var(--border2)}
.sc-id{font-family:var(--m);font-size:9px;color:var(--text3);margin-bottom:5px;letter-spacing:.5px}
.sc-state{font-size:12px;font-weight:600;margin-bottom:10px}
.sc.att .sc-state{color:var(--accent)}.sc.dist .sc-state{color:var(--red)}
.sc-spark{display:flex;gap:2px;align-items:flex-end;height:18px}
.sc-bar{flex:1;border-radius:2px;min-height:2px;transition:height .2s}
.sc-bar.a{background:var(--accent)}
.sc-bar.d{background:var(--red)}
@keyframes slideUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.sc{animation:slideUp .25s var(--ease) both}

/* ── SUMMARY BANNER ── */
#summary-banner{
  display:none;background:var(--surface);
  border:1px solid rgba(255,179,71,.25);border-radius:14px;
  padding:22px 24px;
  animation:slideUp .4s var(--ease);
}
.summary-title{
  font-family:var(--m);font-size:11px;color:var(--amber);
  letter-spacing:1.2px;text-transform:uppercase;margin-bottom:16px;
  display:flex;align-items:center;gap:8px;
}
.sum-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin-bottom:16px}
.sum-item{background:var(--surface2);border-radius:10px;padding:12px 14px}
.sum-label{font-size:9px;text-transform:uppercase;letter-spacing:1px;color:var(--text3);margin-bottom:5px}
.sum-value{font-family:var(--m);font-size:18px;font-weight:600;color:var(--text)}
.sum-sub{font-size:10px;color:var(--text3);margin-top:3px;font-family:var(--m)}

/* ── AVG TABLE ── */
.avg-table{width:100%;border-collapse:collapse}
.avg-table th{
  font-size:9px;text-transform:uppercase;letter-spacing:1px;
  color:var(--text3);text-align:left;padding:0 8px 10px 0;
  border-bottom:1px solid var(--border);font-family:var(--m);
}
.avg-table td{padding:8px 8px 8px 0;border-bottom:1px solid var(--border);font-size:12px}
.avg-table tr:last-child td{border-bottom:none}
.avg-bar-track{width:100px;height:5px;background:var(--surface3);border-radius:3px;display:inline-block;vertical-align:middle;margin-right:8px}
.avg-bar-fill{height:100%;border-radius:3px;transition:width .5s var(--ease)}

/* ── STATUS BAR ── */
.statusbar{
  height:28px;flex-shrink:0;
  background:var(--surface);border-top:1px solid var(--border);
  display:flex;align-items:center;padding:0 20px;gap:16px;
  font-family:var(--m);font-size:10px;color:var(--text3);
}
.status-item{display:flex;align-items:center;gap:5px}
.status-dot{width:5px;height:5px;border-radius:50%;background:var(--text3)}
.status-dot.green{background:var(--accent)}
.status-dot.red{background:var(--red)}
.status-val{color:var(--text2)}

/* ── NOTIFICATION TOAST ── */
.toast{
  position:fixed;bottom:36px;right:20px;z-index:999;
  background:var(--surface2);border:1px solid var(--border2);
  border-radius:10px;padding:10px 14px;
  font-size:12px;color:var(--text);
  display:flex;align-items:center;gap:8px;
  transform:translateY(20px);opacity:0;
  transition:all .3s var(--ease);pointer-events:none;
  max-width:260px;
}
.toast.show{transform:translateY(0);opacity:1}
.toast-icon{width:20px;height:20px;border-radius:6px;display:grid;place-items:center;flex-shrink:0;font-size:11px}

/* ── SEPARATOR ── */
.sep{height:1px;background:var(--border);margin:0 -20px}

/* ── RESPONSIVE ── */
@media(max-width:1100px){.grid-cam-kpi{grid-template-columns:1fr}.grid-kpi-4{grid-template-columns:repeat(2,1fr)}}
@media(max-width:700px){.grid-cols-2,.grid-cols-3{grid-template-columns:1fr}.students-grid{grid-template-columns:repeat(auto-fill,minmax(130px,1fr))}}
</style>
</head>
<body>

<div class="toast" id="toast">
  <div class="toast-icon" id="toast-icon"></div>
  <span id="toast-msg"></span>
</div>

<div class="app">

  <!-- ══ SIDEBAR ══ -->
  <nav class="sidebar">
    <div class="sidebar-logo" onclick="showPage('overview')">
      <svg viewBox="0 0 24 24" fill="none" stroke="#0a0c10" stroke-width="2.5" stroke-linecap="round">
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
      </svg>
    </div>

    <button class="nav-btn active" id="nav-overview" onclick="showPage('overview')" title="Overview">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
    </button>
    <button class="nav-btn" id="nav-camera" onclick="showPage('camera')" title="Live Camera">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>
    </button>
    <button class="nav-btn" id="nav-students" onclick="showPage('students')" title="Students">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/></svg>
    </button>
    <button class="nav-btn" id="nav-log" onclick="showPage('log')" title="Distraction Log">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
    </button>

    <div class="sidebar-spacer"></div>
    <div class="sidebar-bottom">
      <button class="nav-btn" title="Settings">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/></svg>
      </button>
    </div>
  </nav>

  <!-- ══ MAIN AREA ══ -->
  <div class="main">

    <!-- Topbar -->
    <div class="topbar">
      <div class="topbar-left">
        <span class="page-title" id="page-title">Overview</span>
        <span class="breadcrumb" id="page-crumb">/ dashboard</span>
      </div>
      <div class="topbar-right">
        <!-- Camera toggle -->
        <label class="toggle-wrap" title="Toggle camera feed">
          <label class="toggle">
            <input type="checkbox" id="camToggle" checked onchange="toggleCamera()"/>
            <span class="toggle-slider"></span>
          </label>
          <span>Camera</span>
        </label>
        <!-- Stop session button -->
        <button class="btn btn-red btn-sm" id="stop-btn" onclick="stopSession()" title="Stop session and generate analytics">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>
          Stop Session
        </button>
        <!-- Session badge -->
        <div class="badge badge-live" id="session-badge">
          <div class="pulse"></div>LIVE
        </div>
        <!-- Time -->
        <span style="font-family:var(--m);font-size:11px;color:var(--text3)" id="clock"></span>
      </div>
    </div>

    <!-- Content -->
    <div class="content">

      <!-- ── SESSION SUMMARY BANNER ── -->
      <div id="summary-banner">
        <div class="summary-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
          Session Summary
        </div>
        <div class="sum-grid" id="sum-grid"></div>
        <div class="card-label" style="margin-bottom:12px"><div class="label-dot amber"></div>Per-Student Attentiveness</div>
        <div id="avg-student-table"></div>
      </div>

      <!-- ════════ PAGE: OVERVIEW ════════ -->
      <div class="page active" id="page-overview">

        <!-- KPI row -->
        <div class="grid-kpi-4">
          <!-- Gauge KPI -->
          <div class="kpi-card gauge-card" style="--kc:var(--accent)">
            <div class="kpi-label">Live Attention</div>
            <div class="gauge-wrap">
              <svg class="gauge-svg" viewBox="0 0 148 84">
                <path class="g-track" d="M 14 78 A 60 60 0 0 1 134 78"/>
                <path class="g-fill" id="gauge-arc" d="M 14 78 A 60 60 0 0 1 134 78"/>
              </svg>
              <div class="gauge-num" id="gauge-label">0%</div>
            </div>
            <div class="gauge-sub">class attention</div>
            <div class="gauge-splits">
              <div class="gauge-split">
                <div class="gauge-split-val" id="gs-att" style="color:var(--accent)">0</div>
                <div class="gauge-split-label">Attentive</div>
              </div>
              <div style="width:1px;background:var(--border)"></div>
              <div class="gauge-split">
                <div class="gauge-split-val" id="gs-dist" style="color:var(--red)">0</div>
                <div class="gauge-split-label">Distracted</div>
              </div>
            </div>
          </div>
          <div class="kpi-card" style="--kc:var(--blue)">
            <div class="kpi-label">Session Average</div>
            <div class="kpi-value" id="kpi-avg">—</div>
            <div class="kpi-sub">since session start</div>
            <div class="kpi-trend" style="color:var(--blue)" id="kpi-avg-trend">—</div>
          </div>
          <div class="kpi-card" style="--kc:var(--purple)">
            <div class="kpi-label">Students Detected</div>
            <div class="kpi-value" id="kpi-total">0</div>
            <div class="kpi-sub" id="kpi-total-sub">no session active</div>
          </div>
          <div class="kpi-card" style="--kc:var(--amber)">
            <div class="kpi-label">Session</div>
            <div class="kpi-value" style="font-size:20px" id="kpi-start">—</div>
            <div class="kpi-sub" id="kpi-fps">waiting…</div>
          </div>
        </div>

        <!-- Charts row -->
        <div class="grid-cols-2">
          <div class="card">
            <div class="card-header">
              <div class="card-label"><div class="label-dot"></div>Attention % Over Time</div>
            </div>
            <canvas id="tlChart" height="140"></canvas>
          </div>
          <div class="card">
            <div class="card-header">
              <div class="card-label"><div class="label-dot blue"></div>Attentive vs Distracted</div>
            </div>
            <canvas id="donutChart" height="140"></canvas>
          </div>
        </div>

        <!-- Distraction log (compact) -->
        <div class="card">
          <div class="card-header">
            <div class="card-label"><div class="label-dot red"></div>Distraction Events
              <span style="color:var(--text3);font-size:9px;font-weight:400;text-transform:none;letter-spacing:0">— class dropped below 50%</span>
            </div>
            <button class="btn btn-sm" onclick="showPage('log')">View all →</button>
          </div>
          <div id="dist-log-mini"></div>
        </div>

      </div>

      <!-- ════════ PAGE: CAMERA ════════ -->
      <div class="page" id="page-camera">
        <div class="grid-cam-kpi">
          <!-- Camera feed -->
          <div class="camera-card">
            <div class="camera-topbar">
              <div class="camera-label">
                <div class="rec-dot" id="rec-dot"></div>
                Live Feed — ClassWatch Pipeline
              </div>
              <div class="camera-controls">
                <label class="toggle-wrap btn btn-sm">
                  <label class="toggle" style="margin:0">
                    <input type="checkbox" id="camToggle2" checked onchange="syncToggle(this)"/>
                    <span class="toggle-slider"></span>
                  </label>
                  <span>Camera</span>
                </label>
              </div>
            </div>
            <div class="camera-body">
              <img id="videoFeed" src="/video_feed" alt="Live camera feed"/>
              <div class="camera-off-overlay" id="cam-off-overlay">
                <div class="camera-off-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><line x1="1" y1="1" x2="23" y2="23"/><path d="M21 21H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h3m3-3h6l2 3h4a2 2 0 0 1 2 2v9.34m-7.72-2.06A4 4 0 1 1 7.72 7.72"/></svg>
                </div>
                <span style="font-size:13px;color:var(--text3)">Camera feed paused</span>
                <button class="btn btn-accent btn-sm" onclick="document.getElementById('camToggle').click()">Enable Camera</button>
              </div>
            </div>
            <div class="camera-footer">
              <div class="cam-stat">FPS <span class="cam-stat-val" id="cam-fps">—</span></div>
              <div class="cam-stat">Students <span class="cam-stat-val" id="cam-students">—</span></div>
              <div class="cam-stat">Attentive <span class="cam-stat-val" id="cam-att">—</span></div>
              <div class="cam-stat" style="margin-left:auto"><span class="cam-stat-badge">PRIVACY ON</span></div>
              <div class="cam-stat"><span class="cam-stat-badge" style="background:rgba(77,158,255,.1);color:var(--blue)">MJPEG</span></div>
            </div>
          </div>

          <!-- Right: gauge + mini stats -->
          <div style="display:flex;flex-direction:column;gap:12px">
            <div class="kpi-card gauge-card" style="--kc:var(--accent)">
              <div class="kpi-label">Live Attention</div>
              <div class="gauge-wrap">
                <svg class="gauge-svg" viewBox="0 0 148 84">
                  <path class="g-track" d="M 14 78 A 60 60 0 0 1 134 78"/>
                  <path class="g-fill" id="gauge-arc2" d="M 14 78 A 60 60 0 0 1 134 78"/>
                </svg>
                <div class="gauge-num" id="gauge-label2">0%</div>
              </div>
              <div class="gauge-sub">real-time</div>
            </div>
            <div class="kpi-card" style="--kc:var(--blue)">
              <div class="kpi-label">Session Average</div>
              <div class="kpi-value" id="cam-avg">—</div>
              <div class="kpi-sub">since session start</div>
            </div>
            <div class="kpi-card" style="--kc:var(--amber)">
              <div class="kpi-label">Session Started</div>
              <div class="kpi-value" style="font-size:20px" id="cam-start">—</div>
              <div class="kpi-sub" id="cam-fps-label">waiting…</div>
            </div>
          </div>
        </div>
      </div>

      <!-- ════════ PAGE: STUDENTS ════════ -->
      <div class="page" id="page-students">
        <div class="card">
          <div class="card-header">
            <div class="card-label"><div class="label-dot purple"></div>Per-Student Live History</div>
            <span style="font-family:var(--m);font-size:10px;color:var(--text3)" id="student-count-label">0 tracked</span>
          </div>
          <div class="students-grid" id="students-grid">
            <div class="dist-empty"><div class="dist-empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg></div>Waiting for students…</div>
          </div>
        </div>
      </div>

      <!-- ════════ PAGE: LOG ════════ -->
      <div class="page" id="page-log">
        <div class="card">
          <div class="card-header">
            <div class="card-label"><div class="label-dot red"></div>Distraction Event Log</div>
            <span style="font-family:var(--m);font-size:10px;color:var(--text3)" id="dist-count-label">0 events</span>
          </div>
          <div class="dist-list" id="dist-log-full">
            <div class="dist-empty">
              <div class="dist-empty-icon">✓</div>
              No distraction events yet — class is focused
            </div>
          </div>
        </div>
      </div>

    </div><!-- /content -->

    <!-- Status bar -->
    <div class="statusbar">
      <div class="status-item">
        <div class="status-dot" id="conn-dot"></div>
        <span>Socket</span>
        <span class="status-val" id="conn-label">—</span>
      </div>
      <div class="status-item">
        <span>Stream</span>
        <span class="status-val" id="stream-label">—</span>
      </div>
      <div class="status-item">
        <span>Updated</span>
        <span class="status-val" id="last-upd">—</span>
      </div>
      <div style="flex:1"></div>
      <div class="status-item">
        <span style="color:var(--accent)">ClassWatch</span>
        <span>v2.0 · Privacy-Aware Attention Analysis</span>
      </div>
    </div>

  </div><!-- /main -->
</div><!-- /app -->

<script>
// ── CLOCK ────────────────────────────────────────────────────
function updateClock(){
  const n=new Date();
  document.getElementById('clock').textContent=
    n.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'});
}
setInterval(updateClock,1000);updateClock();

// ── TOAST ────────────────────────────────────────────────────
function showToast(msg, icon='ℹ', color='var(--blue)'){
  const t=document.getElementById('toast');
  document.getElementById('toast-icon').textContent=icon;
  document.getElementById('toast-icon').style.background=color+'22';
  document.getElementById('toast-icon').style.color=color;
  document.getElementById('toast-msg').textContent=msg;
  t.classList.add('show');
  clearTimeout(t._to);
  t._to=setTimeout(()=>t.classList.remove('show'),2800);
}

// ── PAGE NAVIGATION ──────────────────────────────────────────
const PAGE_META={
  overview:  {title:'Overview',  crumb:'/ dashboard'},
  camera:    {title:'Live Camera',crumb:'/ feed'},
  students:  {title:'Students',  crumb:'/ per-student'},
  log:       {title:'Log',       crumb:'/ distraction events'},
};
function showPage(id){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-btn[id^=nav-]').forEach(b=>b.classList.remove('active'));
  document.getElementById('page-'+id).classList.add('active');
  const nb=document.getElementById('nav-'+id);
  if(nb) nb.classList.add('active');
  const m=PAGE_META[id]||{title:id,crumb:''};
  document.getElementById('page-title').textContent=m.title;
  document.getElementById('page-crumb').textContent=m.crumb;
}

// ── CAMERA TOGGLE ────────────────────────────────────────────
let camOn=true;
function toggleCamera(){
  camOn=document.getElementById('camToggle').checked;
  document.getElementById('camToggle2').checked=camOn;
  applyCamera();
}
function syncToggle(el){
  camOn=el.checked;
  document.getElementById('camToggle').checked=camOn;
  applyCamera();
}
function applyCamera(){
  const vid=document.getElementById('videoFeed');
  const overlay=document.getElementById('cam-off-overlay');
  const recDot=document.getElementById('rec-dot');
  const badge=document.getElementById('session-badge');
  if(camOn){
    vid.src='/video_feed?t='+Date.now();
    vid.classList.remove('hidden');
    overlay.classList.remove('show');
    recDot.style.background='var(--red)';
    recDot.style.animation='pulse 1.4s ease infinite';
    showToast('Camera feed enabled','▶','#00f5a8');
    fetch('/toggle_camera',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:true})});
  } else {
    vid.classList.add('hidden');
    overlay.classList.add('show');
    recDot.style.background='var(--text3)';
    recDot.style.animation='none';
    showToast('Camera feed paused','⏸','#ffb347');
    fetch('/toggle_camera',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:false})});
  }
}

// ── GAUGE ────────────────────────────────────────────────────
function setGauge(ids,pct){
  ids.forEach(({arc,label})=>{
    const el=document.getElementById(arc);
    const lb=document.getElementById(label);
    if(!el||!lb)return;
    el.style.strokeDashoffset=231-(pct/100)*231;
    const col=pct>75?'#00f5a8':pct>40?'#ffb347':'#ff4d6d';
    el.style.stroke=col;
    lb.textContent=pct+'%';
    lb.style.color=col;
  });
}

// ── CHARTS ───────────────────────────────────────────────────
const CHART_OPTS={
  responsive:true,
  animation:{duration:400},
  plugins:{legend:{display:false}},
  scales:{
    x:{ticks:{color:'#4a5568',maxTicksLimit:7,font:{family:"'JetBrains Mono',monospace",size:9}},grid:{color:'rgba(255,255,255,0.04)'},border:{display:false}},
    y:{min:0,max:100,ticks:{color:'#4a5568',font:{family:"'JetBrains Mono',monospace",size:9},callback:v=>v+'%'},grid:{color:'rgba(255,255,255,0.04)'},border:{display:false}}
  }
};

const tlCtx=document.getElementById('tlChart').getContext('2d');
const tlChart=new Chart(tlCtx,{
  type:'line',
  data:{labels:[],datasets:[{
    label:'Attention %',data:[],
    borderColor:'#00f5a8',
    backgroundColor:ctx=>{const g=ctx.chart.ctx.createLinearGradient(0,0,0,180);g.addColorStop(0,'rgba(0,245,168,0.18)');g.addColorStop(1,'rgba(0,245,168,0)');return g;},
    borderWidth:2,pointRadius:0,tension:.42,fill:true
  }]},
  options:{...CHART_OPTS,animation:{duration:300}}
});

const dnCtx=document.getElementById('donutChart').getContext('2d');
const donut=new Chart(dnCtx,{
  type:'doughnut',
  data:{labels:['Attentive','Distracted'],datasets:[{
    data:[0,1],
    backgroundColor:['#00f5a8','#ff4d6d'],
    borderColor:'#0f1421',borderWidth:4,hoverOffset:5
  }]},
  options:{
    responsive:true,cutout:'72%',animation:{duration:500},
    plugins:{legend:{position:'bottom',labels:{color:'#8b97b0',font:{family:"'JetBrains Mono',monospace",size:10},padding:16,usePointStyle:true,pointStyleWidth:8}}}
  }
});

// ── DISTRACTION LOG RENDER ────────────────────────────────────
let _distEvents=[];
function renderDistLog(log){
  _distEvents=log||[];
  // Mini (overview)
  const mini=document.getElementById('dist-log-mini');
  // Full (log page)
  const full=document.getElementById('dist-log-full');
  document.getElementById('dist-count-label').textContent=_distEvents.length+' events';
  const empty=`<div class="dist-empty"><div class="dist-empty-icon" style="background:rgba(0,245,168,.08);color:var(--accent)">✓</div>No distraction events — class is focused</div>`;
  if(!_distEvents.length){mini.innerHTML=empty;full.innerHTML=empty;return;}
  const rows=_distEvents.map(ev=>{
    const drop=50-ev.pct;const barW=Math.min(100,Math.max(0,(drop/50)*100));
    return`<div class="dist-row">
      <span class="dist-time">⏱ ${ev.time}</span>
      <span class="dist-pct">${ev.pct}%</span>
      <div class="dist-bar-bg"><div class="dist-bar-fg" style="width:${barW}%"></div></div>
    </div>`;
  }).join('');
  mini.innerHTML=rows;full.innerHTML=rows;
}

// ── STUDENT CARDS RENDER ──────────────────────────────────────
function renderStudents(students){
  const grid=document.getElementById('students-grid');
  const count=Object.keys(students||{}).length;
  document.getElementById('student-count-label').textContent=count+' tracked';
  if(!count){
    grid.innerHTML='<div class="dist-empty"><div class="dist-empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg></div>Waiting for students…</div>';
    return;
  }
  const entries=Object.entries(students).sort((a,b)=>+a[0]-+b[0]);
  grid.innerHTML='';
  entries.forEach(([id,info],i)=>{
    const att=info.current==='Attentive';
    const bars=(info.history||[]).map(h=>`<div class="sc-bar ${h==='Attentive'?'a':'d'}" style="height:${h==='Attentive'?100:55}%"></div>`).join('');
    const c=document.createElement('div');
    c.className=`sc ${att?'att':'dist'}`;
    c.style.animationDelay=(i*0.03)+'s';
    c.innerHTML=`<div class="sc-id">STUDENT #${id}</div>
      <div class="sc-state">${info.current}</div>
      <div class="sc-spark">${bars||'<span style="color:var(--text3);font-size:9px">no data</span>'}</div>`;
    grid.appendChild(c);
  });
}

// ── SESSION SUMMARY ───────────────────────────────────────────
function showSummary(s){
  const badge=document.getElementById('session-badge');
  badge.className='badge badge-ended';
  badge.innerHTML='SESSION ENDED';

  const items=[
    ['Average Attention',  s.average_attention+'%',  null],
    ['Peak Attention',     s.max_attention+'%',      '@ '+s.peak_time],
    ['Lowest Attention',   s.min_attention+'%',      '@ '+s.low_time],
    ['Max Single Drop',    s.max_drop+'%',           null],
    ['Stability Score',    s.stability_score+' / 100',null],
    ['Total Students',     s.total_students,         null],
    ['Distraction Events', s.distraction_events,     null],
    ['Log Entries',        s.row_count,              null],
  ];
  document.getElementById('sum-grid').innerHTML=items.map(([l,v,sub])=>`
    <div class="sum-item">
      <div class="sum-label">${l}</div>
      <div class="sum-value">${v}</div>
      ${sub?`<div class="sum-sub">${sub}</div>`:''}
    </div>`).join('');

  const sts=Object.entries(s.student_avgs||{}).sort((a,b)=>b[1]-a[1]);
  if(sts.length){
    let tbl='<table class="avg-table"><thead><tr><th>Student</th><th>Avg Attentiveness</th><th>Score</th></tr></thead><tbody>';
    sts.forEach(([id,avg])=>{
      const col=avg>=75?'#00f5a8':avg>=40?'#ffb347':'#ff4d6d';
      tbl+=`<tr>
        <td style="font-family:var(--m);color:var(--purple)">Student #${id}</td>
        <td><div class="avg-bar-track"><div class="avg-bar-fill" style="width:${avg}%;background:${col}"></div></div><span style="font-family:var(--m);font-size:11px;color:var(--text)">${avg}%</span></td>
        <td style="font-family:var(--m);font-size:11px;color:${col}">${avg>=75?'Good':avg>=40?'Fair':'Low'}</td>
      </tr>`;
    });
    tbl+='</tbody></table>';
    document.getElementById('avg-student-table').innerHTML=tbl;
  }
  const banner=document.getElementById('summary-banner');
  banner.style.display='block';
  setTimeout(()=>banner.scrollIntoView({behavior:'smooth'}),120);
  showToast('Session ended — summary ready','📋','#ffb347');
}

// ── SOCKET ────────────────────────────────────────────────────
const socket=io();
socket.on('connect',()=>{
  document.getElementById('conn-dot').className='status-dot green';
  document.getElementById('conn-label').textContent='connected';
});
socket.on('disconnect',()=>{
  document.getElementById('conn-dot').className='status-dot red';
  document.getElementById('conn-label').textContent='disconnected';
});
socket.on('session_summary',showSummary);
fetch('/summary').then(r=>r.json()).then(d=>{if(d&&d.ready)showSummary(d);});

socket.on('attention_update',d=>{
  // Gauges (both pages)
  setGauge([{arc:'gauge-arc',label:'gauge-label'},{arc:'gauge-arc2',label:'gauge-label2'}],d.pct);
  // KPIs
  document.getElementById('kpi-avg').textContent=d.avg_pct+'%';
  document.getElementById('kpi-total').textContent=d.total;
  document.getElementById('kpi-total-sub').textContent=d.attentive+' attentive now';
  document.getElementById('kpi-start').textContent=d.start_time;
  document.getElementById('kpi-fps').textContent='FPS: '+d.fps;
  document.getElementById('kpi-avg-trend').textContent=d.avg_pct>50?'▲ above threshold':'▼ below threshold';
  // Camera page KPIs
  document.getElementById('cam-avg').textContent=d.avg_pct+'%';
  document.getElementById('cam-start').textContent=d.start_time;
  document.getElementById('cam-fps-label').textContent='FPS: '+d.fps;
  // Camera footer
  document.getElementById('cam-fps').textContent=d.fps;
  document.getElementById('cam-students').textContent=d.total;
  document.getElementById('cam-att').textContent=d.attentive;
  // Gauge splits
  document.getElementById('gs-att').textContent=d.attentive;
  document.getElementById('gs-dist').textContent=d.total-d.attentive;
  // Charts
  tlChart.data.labels=d.timeline.map(t=>t.time);
  tlChart.data.datasets[0].data=d.timeline.map(t=>t.pct);
  tlChart.update('none');
  donut.data.datasets[0].data=[d.attentive,Math.max(0,d.total-d.attentive)];
  donut.update('none');
  // Logs & students
  renderDistLog(d.distraction_log);
  renderStudents(d.students);
  // Status bar
  document.getElementById('last-upd').textContent=new Date().toLocaleTimeString();
});

// ── VIDEO STREAM STATUS ───────────────────────────────────────
const vid=document.getElementById('videoFeed');
vid.onload=()=>{ document.getElementById('stream-label').textContent='live ✓'; };
vid.onerror=()=>{ document.getElementById('stream-label').textContent='no signal'; };

// ── STOP SESSION ─────────────────────────────────────────────
function stopSession(){
  if(!confirm('Stop the session and generate analytics?')) return;
  document.getElementById('stop-btn').disabled=true;
  document.getElementById('stop-btn').textContent='Stopping…';
  fetch('/shutdown',{method:'POST'})
    .then(()=>showToast('Session stopping…','⏹','#ffb347'))
    .catch(()=>showToast('Send q in terminal to stop','⌨','#ffb347'));
}
</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(_HTML)

@app.route("/summary")
def summary_api():
    if _final_summary is None:
        return jsonify({"ready": False})
    return jsonify({**_final_summary, "ready": True})

@app.route("/toggle_camera", methods=["POST"])
def toggle_camera():
    """Toggle the camera feed on/off from the browser."""
    global _cam_enabled
    from flask import request as req
    data = req.get_json(silent=True) or {}
    _cam_enabled = bool(data.get("enabled", True))
    return jsonify({"ok": True, "enabled": _cam_enabled})

# ── Shutdown endpoint (called by browser Stop button or Ctrl-C) ──
_stop_event_ref = None

def register_shutdown(event):
    """Called by main.py to give dashboard a handle to the stop event."""
    global _stop_event_ref
    _stop_event_ref = event

@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Graceful shutdown — stops the main detection loop."""
    if _stop_event_ref is not None:
        _stop_event_ref.set()
    return jsonify({"ok": True})

@app.route("/video_feed")
def video_feed():
    """MJPEG stream — served to <img src='/video_feed'>."""
    # Optional stream password check
    from flask import request as _req
    if STREAM_PASSWORD:
        auth = _req.args.get("pwd", "")
        if auth != STREAM_PASSWORD:
            return Response("Unauthorized", status=401)

    def generate():
        global _last_jpeg
        while True:
            if not _cam_enabled:
                # Send a tiny placeholder frame while paused
                import time; time.sleep(0.1); continue
            try:
                jpeg = _frame_queue.get(timeout=5.0)
                with _frame_lock:
                    _last_jpeg = jpeg
            except queue.Empty:
                with _frame_lock:
                    jpeg = _last_jpeg
                if not jpeg:
                    continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ─────────────────────────────────────────────────────────────
# Public API — called from main.py
# ─────────────────────────────────────────────────────────────

def push_frame(frame):
    """Encode frame as JPEG and push to MJPEG stream. Call instead of cv2.imshow()."""
    if frame is None or frame.size == 0 or not _cam_enabled:
        return
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return
    jpeg = buf.tobytes()
    try:
        _frame_queue.get_nowait()
    except queue.Empty:
        pass
    _frame_queue.put_nowait(jpeg)


def print_live_dashboard(attentive, total, avg_pct, session_start, fps=0.0):
    global _last_was_distracted
    pct = round(attentive / total * 100, 1) if total > 0 else 0.0
    now = datetime.now().strftime("%H:%M:%S")
    is_distracted = pct < DISTRACTION_THRESHOLD

    with _state_lock:
        _state["pct"]        = pct
        _state["avg_pct"]    = avg_pct
        _state["attentive"]  = attentive
        _state["total"]      = total
        _state["fps"]        = round(fps, 1)
        _state["start_time"] = session_start

        _state["timeline"].append({"time": now, "pct": pct})
        if len(_state["timeline"]) > 60:
            _state["timeline"].pop(0)

        if pct > _state["peak_pct"]:
            _state["peak_pct"] = pct; _state["peak_time"] = now
        if total > 0 and pct < _state["low_pct"]:
            _state["low_pct"]  = pct; _state["low_time"]  = now

        if is_distracted and not _last_was_distracted:
            _state["distraction_log"].append({"time": now, "pct": pct})

        payload = {
            "pct": pct, "avg_pct": avg_pct,
            "attentive": attentive, "total": total,
            "fps": round(fps, 1), "start_time": session_start,
            "students":        _state["students"],
            "timeline":        _state["timeline"][-30:],
            "distraction_log": _state["distraction_log"],
        }

    _last_was_distracted = is_distracted
    socketio.emit("attention_update", payload)


def update_student(track_id: int, state: str):
    tid = str(track_id)
    with _state_lock:
        if tid not in _state["students"]:
            _state["students"][tid] = {"current": state, "history": [], "att_count": 0, "total_count": 0}
        s = _state["students"][tid]
        s["current"] = state
        s["history"].append(state)
        if len(s["history"]) > 30: s["history"].pop(0)
        s["total_count"] += 1
        if state == "Attentive": s["att_count"] += 1


def start_web_dashboard():
    def _run():
        socketio.run(app, host=WEB_HOST, port=WEB_PORT,
                     debug=False, use_reloader=False, log_output=False)
    threading.Thread(target=_run, daemon=True).start()
    threading.Timer(1.4, lambda: webbrowser.open(f"http://localhost:{WEB_PORT}")).start()
    print(f"  [dashboard] Web dashboard → http://localhost:{WEB_PORT}")
    print(f"  [dashboard] Video stream  → http://localhost:{WEB_PORT}/video_feed")


def run_final_dashboard():
    global _final_summary
    print_section("POST-SESSION ANALYTICS")
    stats          = compute_statistics(LOG_PATH)
    student_scores = compute_student_scores(STUDENT_LOG_PATH)
    if not stats:
        print("  [dashboard] No data logged."); return
    generate_graph(LOG_PATH, GRAPH_PATH)
    summary_text = generate_summary(stats, student_scores, SUMMARY_PATH)
    print(summary_text)
    with _state_lock:
        student_avgs = {
            tid: round(info["att_count"] / info["total_count"] * 100, 1)
            for tid, info in _state["students"].items()
            if info["total_count"] > 0
        }
        peak_time   = _state["peak_time"]
        low_time    = _state["low_time"]
        dist_events = len(_state["distraction_log"])
    _final_summary = {
        "average_attention":  stats.get("average_attention", 0),
        "max_attention":      stats.get("max_attention", 0),
        "min_attention":      stats.get("min_attention", 0),
        "max_drop":           stats.get("max_drop", 0),
        "stability_score":    stats.get("stability_score", 0),
        "total_students":     stats.get("total_students", 0),
        "row_count":          stats.get("row_count", 0),
        "distraction_events": dist_events,
        "peak_time":          peak_time,
        "low_time":           low_time,
        "student_avgs":       student_avgs,
    }
    socketio.emit("session_summary", _final_summary)
    print_section("OUTPUT FILES")
    print(f"  {'Graph':<20} →  {GRAPH_PATH}")
    print(f"  {'Summary':<20} →  {SUMMARY_PATH}")
    print(f"  {'Attention Log':<20} →  {LOG_PATH}")
    print(f"  {'Student Log':<20} →  {STUDENT_LOG_PATH}\n")