# =========================================================
# logger.py — CLASS-BASED ATTENTION LOGGER (FINAL FIX)
# =========================================================

import csv
import os
import time
from datetime import datetime

class AttentionLogger:

    def __init__(self, interval=3):
        self.interval = interval
        self.last_log_time = time.time()
        self.log_file = "data/attention_log.csv"

        self._ensure_file()

    # -------------------------------
    # CREATE FILE IF NOT EXISTS
    # -------------------------------
    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "attention_percentage",
                    "total_students",
                    "attentive_count",
                    "distracted_count"
                ])

    # -------------------------------
    # LOG CLASS ATTENTION
    # -------------------------------
    def log_attention(self, data):

        current_time = time.time()

        if current_time - self.last_log_time < self.interval:
            return False

        total = data.get("total_students", 0)
        attentive = data.get("attentive_count", 0)
        distracted = total - attentive

        percentage = (attentive / total * 100) if total > 0 else 0

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round(percentage, 2),
                total,
                attentive,
                distracted
            ])

        self.last_log_time = current_time
        return True

    # -------------------------------
    # LOG STUDENT DATA
    # -------------------------------
    def log_student_state(self, path, student_id, state):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                student_id,
                state
            ])