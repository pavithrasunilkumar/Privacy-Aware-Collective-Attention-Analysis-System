# ============================================================
# dashboard.py — Web Dashboard (Flask + SocketIO)
# Changes vs previous version:
#   • Removed top5/bot5 student lists
#   • Summary shows per-student average attentiveness table
#   • Summary shows exact time of peak and lowest attention
#   • /summary route so browser always shows final summary
#     even if the tab was opened after 'q' was pressed
# ============================================================

import threading
import webbrowser
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO
from analytics import (compute_statistics, compute_student_scores,
                        generate_graph, generate_summary)
from utils import print_section

LOG_PATH         = "data/attention_log.csv"
STUDENT_LOG_PATH = "data/student_log.csv"
GRAPH_PATH       = "outputs/attention_graph.png"
SUMMARY_PATH     = "outputs/summary.txt"
WEB_PORT         = 5000
DISTRACTION_THRESHOLD = 50.0   # % — below this = distraction event

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

_state = {
    "pct": 0.0, "avg_pct": 0.0,
    "attentive": 0, "total": 0,
    "fps": 0.0, "start_time": "—",
    "students": {},        # { "id": {"current", "history", "att_count", "total_count"} }
    "timeline": [],        # [{"time", "pct"}, ...]
    "distraction_log": [], # [{"time", "pct"}, ...]
    # track running peak/low with their timestamps
    "peak_pct": 0.0,   "peak_time": "—",
    "low_pct":  100.0, "low_time":  "—",
}
_last_was_distracted = False
_state_lock = threading.Lock()

# stored after session ends — served to late-joiners via /summary
_final_summary = None

# ─────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ClassWatch · Live Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0c10;--surface:#111318;--surface2:#181b22;--border:#1f2430;
  --accent:#00e5a0;--red:#ff4b6e;--yellow:#fbbf24;--purple:#818cf8;
  --text:#e8eaf0;--muted:#5a6078;
  --mono:'Space Mono',monospace;--sans:'DM Sans',sans-serif;
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");opacity:.4}
.shell{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:0 24px 48px}

/* header */
header{display:flex;align-items:center;justify-content:space-between;padding:28px 0 24px;border-bottom:1px solid var(--border);margin-bottom:32px}
.logo{display:flex;align-items:center;gap:12px}
.logo-icon{width:36px;height:36px;background:var(--accent);border-radius:8px;display:grid;place-items:center}
.logo-text{font-family:var(--mono);font-size:17px;font-weight:700;letter-spacing:-.5px}
.logo-text span{color:var(--accent)}
.live-badge{display:flex;align-items:center;gap:6px;background:rgba(0,229,160,.08);border:1px solid rgba(0,229,160,.2);border-radius:999px;padding:5px 14px;font-family:var(--mono);font-size:11px;color:var(--accent);letter-spacing:1px;text-transform:uppercase;transition:all .4s}
.live-badge.ended{background:rgba(251,191,36,.08);border-color:rgba(251,191,36,.3);color:var(--yellow)}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--accent);animation:pulse 1.4s ease infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}

/* summary banner */
#summary-banner{display:none;background:var(--surface);border:1px solid var(--yellow);border-radius:12px;padding:24px 28px;margin-bottom:28px;animation:fadeIn .5s ease}
#summary-banner h2{font-family:var(--mono);font-size:13px;color:var(--yellow);letter-spacing:1px;text-transform:uppercase;margin-bottom:18px}
.sum-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;margin-bottom:20px}
.sum-item{background:var(--surface2);border-radius:8px;padding:12px 16px}
.s-label{font-size:10px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:4px}
.s-val{font-family:var(--mono);font-size:20px;font-weight:700;color:var(--text)}
.s-sub{font-size:11px;color:var(--muted);margin-top:3px;font-family:var(--mono)}

/* student avg table */
.avg-table{width:100%;border-collapse:collapse;margin-top:4px}
.avg-table th{font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);text-align:left;padding:0 12px 10px 0;border-bottom:1px solid var(--border)}
.avg-table td{padding:8px 12px 8px 0;border-bottom:1px solid var(--border);font-size:13px}
.avg-table td:first-child{font-family:var(--mono);font-size:12px;color:var(--purple)}
.avg-bar-wrap{width:120px;height:6px;background:var(--border);border-radius:3px;display:inline-block;vertical-align:middle;margin-right:8px}
.avg-bar{height:100%;border-radius:3px;transition:width .4s}

/* kpi row */
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 22px;position:relative;overflow:hidden}
.kpi::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--kpi-color,var(--accent));opacity:.7}
.kpi:nth-child(2){--kpi-color:var(--yellow)}
.kpi:nth-child(3){--kpi-color:var(--purple)}
.kpi:nth-child(4){--kpi-color:var(--red)}
.kpi-label{font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:10px}
.kpi-value{font-family:var(--mono);font-size:32px;font-weight:700;line-height:1;transition:color .3s}
.kpi-sub{font-size:12px;color:var(--muted);margin-top:6px}

/* gauge */
.gauge-wrap{position:relative;width:140px;height:80px;margin:4px 0 0}
.gauge-svg{width:140px;height:80px;overflow:visible}
.gauge-track{fill:none;stroke:var(--border);stroke-width:12;stroke-linecap:round}
.gauge-fill{fill:none;stroke:var(--accent);stroke-width:12;stroke-linecap:round;stroke-dasharray:220;stroke-dashoffset:220;transition:stroke-dashoffset .6s ease,stroke .4s}
.gauge-label{position:absolute;bottom:0;left:50%;transform:translateX(-50%);font-family:var(--mono);font-size:22px;font-weight:700;white-space:nowrap}

/* charts */
.charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:22px 24px;margin-bottom:24px}
.card-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:18px;display:flex;align-items:center;gap:8px}
.dot{width:6px;height:6px;border-radius:50%;background:var(--accent)}
.dot.p{background:var(--purple)}.dot.r{background:var(--red)}.dot.y{background:var(--yellow)}

/* distraction log */
.dist-table{width:100%;border-collapse:collapse}
.dist-table th{font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);text-align:left;padding:0 0 10px;border-bottom:1px solid var(--border)}
.dist-table td{padding:9px 0;border-bottom:1px solid var(--border);font-size:13px;vertical-align:middle}
.dist-table td:first-child{font-family:var(--mono);font-size:12px;color:var(--yellow);width:110px}
.dist-pill{display:inline-flex;align-items:center;gap:5px;background:rgba(255,75,110,.1);border:1px solid rgba(255,75,110,.25);border-radius:999px;padding:2px 10px;font-family:var(--mono);font-size:11px;color:var(--red)}
.empty-log{color:var(--muted);font-size:13px;padding:12px 0}

/* students */
.students-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:10px}
.sc{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:14px 16px;animation:fadeIn .3s ease;transition:border-color .3s}
.sc.att{border-color:rgba(0,229,160,.3)}.sc.dist{border-color:rgba(255,75,110,.3)}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.sc-id{font-family:var(--mono);font-size:11px;color:var(--muted);margin-bottom:4px}
.sc-state{font-size:13px;font-weight:600;margin-bottom:8px}
.sc.att .sc-state{color:var(--accent)}.sc.dist .sc-state{color:var(--red)}
.spark{display:flex;gap:2px;align-items:flex-end;height:18px}
.sb{flex:1;border-radius:2px;min-height:3px}
.sb.a{background:var(--accent);height:100%}.sb.d{background:var(--red);height:60%}

/* status */
.status-bar{margin-top:20px;display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;color:var(--muted)}
#conn{padding:2px 8px;border-radius:4px;background:rgba(255,75,110,.1);color:var(--red);transition:all .3s}
#conn.on{background:rgba(0,229,160,.1);color:var(--accent)}

@media(max-width:900px){.kpi-row,.charts-grid{grid-template-columns:1fr 1fr}}
@media(max-width:560px){.kpi-row{grid-template-columns:1fr}header{flex-direction:column;gap:14px;align-items:flex-start}}
</style>
</head>
<body>
<div class="shell">

  <header>
    <div class="logo">
      <div class="logo-icon"><svg viewBox="0 0 24 24" fill="none" stroke="#0a0c10" stroke-width="2.5" stroke-linecap="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg></div>
      <div class="logo-text">Class<span>Watch</span></div>
    </div>
    <div class="live-badge" id="live-badge"><div class="live-dot" id="live-dot"></div><span id="badge-text">LIVE</span></div>
  </header>

  <!-- Session-end summary banner -->
  <div id="summary-banner">
    <h2>📋 Session Summary</h2>
    <div class="sum-grid" id="sum-grid"></div>
    <div class="card-title" style="margin-top:4px"><div class="dot y"></div>Per-Student Average Attentiveness</div>
    <div id="avg-student-table"></div>
  </div>

  <!-- KPI row -->
  <div class="kpi-row">
    <div class="kpi">
      <div class="kpi-label">Live Attention</div>
      <div class="gauge-wrap">
        <svg class="gauge-svg" viewBox="0 0 140 80">
          <path class="gauge-track" d="M 14 74 A 56 56 0 0 1 126 74"/>
          <path class="gauge-fill" id="gauge-arc" d="M 14 74 A 56 56 0 0 1 126 74"/>
        </svg>
        <div class="gauge-label" id="gauge-label">0%</div>
      </div>
    </div>
    <div class="kpi"><div class="kpi-label">Session Average</div><div class="kpi-value" id="avg-pct">—</div><div class="kpi-sub">since session start</div></div>
    <div class="kpi"><div class="kpi-label">Students Detected</div><div class="kpi-value" id="total">0</div><div class="kpi-sub"><span id="att-count">0</span> attentive now</div></div>
    <div class="kpi"><div class="kpi-label">Session Started</div><div class="kpi-value" style="font-size:22px" id="start-time">—</div><div class="kpi-sub" id="fps-label">waiting…</div></div>
  </div>

  <!-- Charts -->
  <div class="charts-grid">
    <div class="card" style="margin-bottom:0"><div class="card-title"><div class="dot"></div>Attention % Over Time</div><canvas id="tlChart" height="180"></canvas></div>
    <div class="card" style="margin-bottom:0"><div class="card-title"><div class="dot p"></div>Attentive vs Distracted</div><canvas id="donutChart" height="180"></canvas></div>
  </div>
  <div style="margin-bottom:24px"></div>

  <!-- Distraction log -->
  <div class="card">
    <div class="card-title"><div class="dot r"></div>Distraction Events
      <span style="font-size:10px;color:var(--muted);margin-left:4px">(attention dropped below 50%)</span>
    </div>
    <div id="dist-log-body"><div class="empty-log">No distraction events yet — class is focused 👍</div></div>
  </div>

  <!-- Per-student cards -->
  <div class="card">
    <div class="card-title"><div class="dot y"></div>Per-Student Live History</div>
    <div class="students-grid" id="students"><div style="color:var(--muted);font-size:13px;padding:8px 0">Waiting for students…</div></div>
  </div>

  <div class="status-bar">Socket: <span id="conn">disconnected</span> &nbsp;·&nbsp; Last update: <span id="last-upd">—</span></div>
</div>

<script>
const socket = io();
const connEl = document.getElementById('conn');
socket.on('connect',    () => { connEl.textContent='connected';    connEl.classList.add('on'); });
socket.on('disconnect', () => { connEl.textContent='disconnected'; connEl.classList.remove('on'); });

// ── Gauge ────────────────────────────────────────────────────
const arc = document.getElementById('gauge-arc');
const glabel = document.getElementById('gauge-label');
function setGauge(p) {
  arc.style.strokeDashoffset = 220 - (p / 100) * 220;
  glabel.textContent = p + '%';
  arc.style.stroke = p > 75 ? '#00e5a0' : p > 40 ? '#fbbf24' : '#ff4b6e';
}

// ── Timeline chart ───────────────────────────────────────────
const tlCtx = document.getElementById('tlChart').getContext('2d');
const tlChart = new Chart(tlCtx, {
  type: 'line',
  data: { labels: [], datasets: [{
    label: 'Attention %', data: [],
    borderColor: '#00e5a0', backgroundColor: 'rgba(0,229,160,.07)',
    borderWidth: 2, pointRadius: 0, tension: .45, fill: true
  }]},
  options: { responsive: true, animation: { duration: 300 },
    scales: {
      x: { ticks: { color:'#5a6078', maxTicksLimit:6, font:{family:'Space Mono',size:10} }, grid:{color:'#1f2430'} },
      y: { min:0, max:100, ticks:{ color:'#5a6078', font:{family:'Space Mono',size:10}, callback:v=>v+'%' }, grid:{color:'#1f2430'} }
    },
    plugins: { legend: { display:false } }
  }
});

// ── Donut chart ──────────────────────────────────────────────
const dnCtx = document.getElementById('donutChart').getContext('2d');
const donut = new Chart(dnCtx, {
  type: 'doughnut',
  data: { labels:['Attentive','Distracted'], datasets:[{data:[0,0],backgroundColor:['#00e5a0','#ff4b6e'],borderColor:'#111318',borderWidth:3,hoverOffset:6}] },
  options: { responsive:true, cutout:'68%', animation:{duration:400},
    plugins:{ legend:{ position:'bottom', labels:{color:'#e6edf3',font:{family:'DM Sans',size:12},padding:16,usePointStyle:true,pointStyleWidth:8} } }
  }
});

// ── Distraction log ──────────────────────────────────────────
function renderDistractionLog(log) {
  const el = document.getElementById('dist-log-body');
  if (!log || !log.length) {
    el.innerHTML = '<div class="empty-log">No distraction events yet — class is focused 👍</div>';
    return;
  }
  let html = '<table class="dist-table"><thead><tr><th>Time</th><th>Attention</th><th>Status</th></tr></thead><tbody>';
  for (const ev of log) {
    html += `<tr><td>${ev.time}</td><td style="padding-right:20px">${ev.pct}%</td><td><span class="dist-pill">⚠ Distracted</span></td></tr>`;
  }
  el.innerHTML = html + '</tbody></table>';
}

// ── Student cards ────────────────────────────────────────────
function renderStudents(students) {
  const grid = document.getElementById('students');
  if (!students || !Object.keys(students).length) {
    grid.innerHTML = '<div style="color:var(--muted);font-size:13px;padding:8px 0">No students detected…</div>';
    return;
  }
  const entries = Object.entries(students).sort((a, b) => +a[0] - +b[0]);
  grid.innerHTML = '';
  for (const [id, info] of entries) {
    const isAtt = info.current === 'Attentive';
    const bars = (info.history || []).map(h => `<div class="sb ${h==='Attentive'?'a':'d'}"></div>`).join('');
    const c = document.createElement('div');
    c.className = `sc ${isAtt ? 'att' : 'dist'}`;
    c.innerHTML = `<div class="sc-id">Student #${id}</div>
      <div class="sc-state">${info.current}</div>
      <div class="spark">${bars || '<span style="color:var(--muted);font-size:10px">—</span>'}</div>`;
    grid.appendChild(c);
  }
}

// ── Session summary banner ───────────────────────────────────
function showSummary(s) {
  // Flip badge
  const badge = document.getElementById('live-badge');
  badge.classList.add('ended');
  document.getElementById('live-dot').style.animation = 'none';
  document.getElementById('badge-text').textContent = 'SESSION ENDED';

  // Stats grid — includes peak/low times
  const items = [
    ['Average Attention',   s.average_attention + '%',  null],
    ['Peak Attention',      s.max_attention + '%',      '@ ' + s.peak_time],
    ['Lowest Attention',    s.min_attention + '%',      '@ ' + s.low_time],
    ['Max Single Drop',     s.max_drop + '%',           null],
    ['Stability Score',     s.stability_score + ' / 100', null],
    ['Total Students',      s.total_students,           null],
    ['Distraction Events',  s.distraction_events,       null],
    ['Log Entries',         s.row_count,                null],
  ];
  document.getElementById('sum-grid').innerHTML = items.map(([l, v, sub]) => `
    <div class="sum-item">
      <div class="s-label">${l}</div>
      <div class="s-val">${v}</div>
      ${sub ? `<div class="s-sub">${sub}</div>` : ''}
    </div>`).join('');

  // Per-student average attentiveness table (sorted desc)
  const students = Object.entries(s.student_avgs || {})
    .sort((a, b) => b[1] - a[1]);

  if (students.length) {
    let tbl = '<table class="avg-table"><thead><tr><th>Student</th><th>Avg Attentiveness</th><th>Score</th></tr></thead><tbody>';
    for (const [id, avg] of students) {
      const colour = avg >= 75 ? '#00e5a0' : avg >= 40 ? '#fbbf24' : '#ff4b6e';
      tbl += `<tr>
        <td>Student #${id}</td>
        <td>
          <div class="avg-bar-wrap"><div class="avg-bar" style="width:${avg}%;background:${colour}"></div></div>
          <span style="font-family:var(--mono);font-size:12px">${avg}%</span>
        </td>
        <td style="font-family:var(--mono);font-size:12px;color:${colour}">${avg >= 75 ? 'Good' : avg >= 40 ? 'Fair' : 'Low'}</td>
      </tr>`;
    }
    tbl += '</tbody></table>';
    document.getElementById('avg-student-table').innerHTML = tbl;
  } else {
    document.getElementById('avg-student-table').innerHTML =
      '<div style="color:var(--muted);font-size:13px;padding:8px 0">No student data recorded.</div>';
  }

  const banner = document.getElementById('summary-banner');
  banner.style.display = 'block';
  setTimeout(() => banner.scrollIntoView({ behavior: 'smooth' }), 100);
}

socket.on('session_summary', showSummary);

// On page load check if session already ended (late joiner)
fetch('/summary').then(r => r.json()).then(d => { if (d && d.ready) showSummary(d); });

// ── Main live update ─────────────────────────────────────────
socket.on('attention_update', d => {
  setGauge(d.pct);
  document.getElementById('avg-pct').textContent    = d.avg_pct + '%';
  document.getElementById('total').textContent      = d.total;
  document.getElementById('att-count').textContent  = d.attentive;
  document.getElementById('start-time').textContent = d.start_time;
  document.getElementById('fps-label').textContent  = 'FPS: ' + d.fps;
  tlChart.data.labels = d.timeline.map(t => t.time);
  tlChart.data.datasets[0].data = d.timeline.map(t => t.pct);
  tlChart.update('none');
  donut.data.datasets[0].data = [d.attentive, d.total - d.attentive];
  donut.update('none');
  renderDistractionLog(d.distraction_log);
  renderStudents(d.students);
  document.getElementById('last-upd').textContent = new Date().toLocaleTimeString();
});
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
    """Returns final summary JSON so late-joining browser tabs can show it."""
    if _final_summary is None:
        return jsonify({"ready": False})
    return jsonify({**_final_summary, "ready": True})


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def print_live_dashboard(attentive, total, avg_pct, session_start, fps=0.0):
    """Push live frame data to browser. Track running peak/low with timestamps."""
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

        # Track peak and low with exact timestamps
        if pct > _state["peak_pct"]:
            _state["peak_pct"]  = pct
            _state["peak_time"] = now
        if total > 0 and pct < _state["low_pct"]:
            _state["low_pct"]  = pct
            _state["low_time"] = now

        # Distraction log — leading edge only
        if is_distracted and not _last_was_distracted:
            _state["distraction_log"].append({"time": now, "pct": pct})

        payload = {
            "pct":             pct,
            "avg_pct":         avg_pct,
            "attentive":       attentive,
            "total":           total,
            "fps":             round(fps, 1),
            "start_time":      session_start,
            "students":        _state["students"],
            "timeline":        _state["timeline"][-30:],
            "distraction_log": _state["distraction_log"],
        }

    _last_was_distracted = is_distracted
    socketio.emit("attention_update", payload)


def update_student(track_id: int, state: str):
    """Update per-student live history + running attentiveness count."""
    tid = str(track_id)
    with _state_lock:
        if tid not in _state["students"]:
            _state["students"][tid] = {
                "current": state, "history": [],
                "att_count": 0, "total_count": 0
            }
        s = _state["students"][tid]
        s["current"] = state
        s["history"].append(state)
        if len(s["history"]) > 30:
            s["history"].pop(0)
        s["total_count"] += 1
        if state == "Attentive":
            s["att_count"] += 1


def start_web_dashboard():
    """Start Flask+SocketIO in background thread and open browser."""
    def _run():
        socketio.run(app, host="0.0.0.0", port=WEB_PORT,
                     debug=False, use_reloader=False, log_output=False)

    threading.Thread(target=_run, daemon=True).start()
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{WEB_PORT}")).start()
    print(f"  [dashboard] Web dashboard → http://localhost:{WEB_PORT}")


# ─────────────────────────────────────────────────────────────
# Post-session
# ─────────────────────────────────────────────────────────────

def run_final_dashboard():
    """Compute analytics, push summary to browser, print to console."""
    global _final_summary
    print_section("POST-SESSION ANALYTICS")

    stats          = compute_statistics(LOG_PATH)
    student_scores = compute_student_scores(STUDENT_LOG_PATH)

    if not stats:
        print("  [dashboard] No data logged.")
        return

    generate_graph(LOG_PATH, GRAPH_PATH)
    summary_text = generate_summary(stats, student_scores, SUMMARY_PATH)
    print(summary_text)

    # Build per-student average attentiveness from live counters
    # (more accurate than CSV-based since it tracks every frame)
    with _state_lock:
        student_avgs = {
            tid: round(info["att_count"] / info["total_count"] * 100, 1)
            for tid, info in _state["students"].items()
            if info["total_count"] > 0
        }
        peak_time = _state["peak_time"]
        low_time  = _state["low_time"]
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