# ============================================================
# analytics.py — Analytics Engine
# ============================================================

import os, csv, math
from datetime import datetime

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MPL = True
except ImportError:
    MPL = False
    print("[analytics] matplotlib not installed — graph skipped.")

LOG_PATH         = "data/attention_log.csv"
STUDENT_LOG_PATH = "data/student_log.csv"
GRAPH_PATH       = "outputs/attention_graph.png"
SUMMARY_PATH     = "outputs/summary.txt"


def _read_log(path=LOG_PATH):
    if not os.path.exists(path): return []
    rows=[]
    with open(path,newline="") as f:
        for r in csv.DictReader(f):
            try:
                r["attention_percentage"]=float(r["attention_percentage"])
                r["total_students"]=int(r["total_students"])
                r["attentive_count"]=int(r["attentive_count"])
                r["distracted_count"]=int(r["distracted_count"])
                rows.append(r)
            except: continue
    return rows


def _read_student_log(path=STUDENT_LOG_PATH):
    if not os.path.exists(path): return {}
    hist={}
    with open(path,newline="") as f:
        for r in csv.DictReader(f):
            hist.setdefault(r.get("track_id","?"),[]).append(r.get("state","Unknown"))
    return hist


def compute_statistics(log_path=LOG_PATH):
    rows=_read_log(log_path)
    if not rows: return {}
    pcts=[r["attention_percentage"] for r in rows]
    avg=round(sum(pcts)/len(pcts),1)
    max_att=round(max(pcts),1); min_att=round(min(pcts),1)
    max_total=max(r["total_students"] for r in rows)
    drops=[pcts[i]-pcts[i+1] for i in range(len(pcts)-1)]
    max_drop=round(max(drops),1) if drops else 0.0
    mean=sum(pcts)/len(pcts)
    stdev=math.sqrt(sum((p-mean)**2 for p in pcts)/len(pcts)) if len(pcts)>1 else 0
    stability=round(max(0.0,100.0-stdev),1)
    return {"average_attention":avg,"max_attention":max_att,"min_attention":min_att,
            "max_drop":max_drop,"stability_score":stability,
            "total_students":max_total,"row_count":len(rows)}


def compute_student_scores(path=STUDENT_LOG_PATH):
    hist=_read_student_log(path)
    return {tid:round(sum(1 for s in states if s=="Attentive")/len(states)*100,1)
            for tid,states in hist.items() if states}


def generate_graph(log_path=LOG_PATH, graph_path=GRAPH_PATH):
    if not MPL: return False
    rows=_read_log(log_path)
    if len(rows)<2: print("[analytics] Need ≥2 data points."); return False
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    timestamps=[r["timestamp"] for r in rows]
    pcts=[r["attention_percentage"] for r in rows]
    try: times=[datetime.strptime(t,"%Y-%m-%d %H:%M:%S") for t in timestamps]
    except: times=list(range(len(pcts)))
    max_pct=max(pcts); min_pct=min(pcts)
    max_i=pcts.index(max_pct); min_i=pcts.index(min_pct)
    fig,ax=plt.subplots(figsize=(12,5))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#161b22")
    ax.plot(times,pcts,color="#00e5a0",linewidth=2,label="Attention %",zorder=3)
    ax.fill_between(times,pcts,alpha=0.12,color="#00e5a0")
    ax.scatter([times[max_i]],[max_pct],color="#fbbf24",s=80,zorder=5,label=f"Peak {max_pct}%")
    ax.annotate(f"Peak\n{max_pct}%",xy=(times[max_i],max_pct),
                xytext=(10,10),textcoords="offset points",color="#fbbf24",fontsize=8)
    ax.scatter([times[min_i]],[min_pct],color="#ff4b6e",s=80,zorder=5,label=f"Low {min_pct}%")
    ax.annotate(f"Low\n{min_pct}%",xy=(times[min_i],min_pct),
                xytext=(10,-20),textcoords="offset points",color="#ff4b6e",fontsize=8)
    avg=sum(pcts)/len(pcts)
    ax.axhline(avg,color="#818cf8",linewidth=1,linestyle="--",label=f"Avg {avg:.1f}%")
    ax.axhline(75,color="#22c55e",linewidth=0.6,linestyle=":",alpha=0.5,label="75% threshold")
    ax.set_ylim(0,105)
    ax.set_xlabel("Time",color="#8b949e",fontsize=10)
    ax.set_ylabel("Attention %",color="#8b949e",fontsize=10)
    ax.set_title("Class Attention Over Time",color="#e6edf3",fontsize=13,fontweight="bold",pad=14)
    ax.tick_params(colors="#8b949e"); ax.spines[:].set_color("#30363d")
    if isinstance(times[0],datetime):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate(rotation=30)
    ax.legend(facecolor="#21262d",labelcolor="#e6edf3",fontsize=9,framealpha=0.8)
    ax.grid(axis="y",color="#21262d",linewidth=0.8)
    plt.tight_layout()
    plt.savefig(graph_path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"[analytics] Graph saved → {graph_path}"); return True


def generate_summary(stats, student_scores, summary_path=SUMMARY_PATH):
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    sorted_s=sorted(student_scores.items(),key=lambda x:x[1],reverse=True)
    top5=sorted_s[:5]; bottom5=sorted_s[-5:][::-1]
    lines=["="*52,"     CLASS ATTENTION SESSION SUMMARY","="*52,
           f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}","",
           "── OVERALL STATS ─────────────────────────────────",
           f"  Average Attention    : {stats.get('average_attention','—')}%",
           f"  Peak Attention       : {stats.get('max_attention','—')}%",
           f"  Lowest Attention     : {stats.get('min_attention','—')}%",
           f"  Max Single Drop      : {stats.get('max_drop','—')}%",
           f"  Stability Score      : {stats.get('stability_score','—')} / 100",
           f"  Total Students Seen  : {stats.get('total_students','—')}",
           f"  Log Entries          : {stats.get('row_count','—')}","",
           "── TOP 5 MOST ATTENTIVE ──────────────────────────"]
    for i,(tid,sc) in enumerate(top5,1): lines.append(f"  #{i}  Student ID {tid:>4}  →  {sc}%")
    lines+=["","── TOP 5 LEAST ATTENTIVE ─────────────────────────"]
    for i,(tid,sc) in enumerate(bottom5,1): lines.append(f"  #{i}  Student ID {tid:>4}  →  {sc}%")
    lines+=["","="*52,""]
    content="\n".join(lines)
    with open(summary_path,"w") as f: f.write(content)
    print(f"[analytics] Summary saved → {summary_path}")
    return content


def run_full_analytics():
    print("\n[analytics] Running post-session analytics…")
    stats=compute_statistics(); student_scores=compute_student_scores()
    generate_graph(); generate_summary(stats,student_scores); 
