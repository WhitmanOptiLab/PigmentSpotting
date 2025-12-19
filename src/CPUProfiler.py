import psutil
import os
import time
from datetime import datetime
import cv2
#import GPUtil
class CPUProfiler:
    def __init__(self, label="block", logfile="cpu_profile.log"):
        self.label = label
        self.logfile = logfile
        self.process = psutil.Process(os.getpid())

    def __enter__(self):
        self.start_wall = time.perf_counter()
        self.start_cpu = self.process.cpu_times()
        self.start_threads = self.process.num_threads()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_wall = time.perf_counter()
        end_cpu = self.process.cpu_times()
        end_threads = self.process.num_threads()

        wall_time = end_wall - self.start_wall
        user_cpu = end_cpu.user - self.start_cpu.user
        sys_cpu = end_cpu.system - self.start_cpu.system
        total_cpu = user_cpu + sys_cpu
        cpu_percent = (total_cpu / wall_time) * 100 if wall_time > 0 else 0.0

        # OpenCV internal threading (can be -1 = default)
        try:
            cv_threads = cv2.getNumThreads()
        except Exception:
            cv_threads = -1

        timestamp = datetime.now().isoformat(timespec="seconds")

        log_line = (
            f"{timestamp},"
            f"{self.label},"
            f"wall={wall_time:.6f},"
            f"user={user_cpu:.6f},"
            f"sys={sys_cpu:.6f},"
            f"cpu={cpu_percent:.2f},"
            f"threads_start={self.start_threads},"
            f"threads_end={end_threads},"
            f"cv_threads={cv_threads}\n"
        )

        with open(self.logfile, "a", encoding="utf-8") as f:
            f.write(log_line)