# ============================================================
# utils.py — Shared Helper Utilities
# ============================================================

import os, cv2, time, numpy as np
from datetime import datetime
from collections import deque


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


class FPSCounter:
    def __init__(self, window=30):
        self._times = deque(maxlen=window)

    def tick(self):
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        return round((len(self._times)-1)/(self._times[-1]-self._times[0]), 1)


def draw_hud(frame, attentive, total, fps, privacy_on=True):
    if frame is None or frame.size == 0:
        return frame
    pct = round(attentive/total*100, 1) if total > 0 else 0.0
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,50), (10,10,15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    colour = (0,220,130) if pct>75 else (0,210,250) if pct>40 else (60,60,255)
    cv2.putText(frame, f"Attention: {attentive}/{total}  ({pct}%)",
                (14,32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
    cv2.putText(frame, f"FPS: {fps}", (w-110,32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
    if privacy_on:
        cv2.putText(frame, "PRIVACY ON", (w-230,32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,100), 1)
    return frame


def draw_person_box(frame, x1, y1, x2, y2, track_id, attention):
    colour = (0,220,130) if attention=="Attentive" else (60,60,255)
    cv2.rectangle(frame, (x1,y1), (x2,y2), colour, 2)
    bg_y = max(0, y1-28)
    cv2.rectangle(frame, (x1,bg_y), (x1+160,y1), colour, -1)
    cv2.putText(frame, f"#{track_id} {attention}", (x1+4, y1-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10,10,10), 2)
    return frame


def print_section(title, width=52):
    print("\n"+"─"*width)
    print(f"  {title}")
    print("─"*width)


def print_live_status(attentive, total, fps):
    pct = round(attentive/total*100,1) if total>0 else 0.0
    bar = "█"*int(pct/5)+"░"*(20-int(pct/5))
    print(f"\r  [{bar}] {pct:5.1f}%  Attentive:{attentive}/{total}  FPS:{fps}   ",
          end="", flush=True)


def timestamp_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_crop(frame, x1, y1, x2, y2, pad=0):
    h,w = frame.shape[:2]
    x1=max(0,x1-pad); y1=max(0,y1-pad)
    x2=min(w,x2+pad); y2=min(h,y2+pad)
    if x2<=x1 or y2<=y1: return None
    return frame[y1:y2,x1:x2]
