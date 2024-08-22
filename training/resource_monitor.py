import psutil
import GPUtil
import threading
import time


class ResourceMonitor(threading.Thread):
    def __init__(self, logger, interval=60):
        threading.Thread.__init__(self)
        self.logger = logger
        self.interval = interval
        self.stopped = threading.Event()

    def run(self):
        while not self.stopped.wait(self.interval):
            cpu_percent = psutil.cpu_percent()
            ram_percent = psutil.virtual_memory().percent
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_load = f"{gpu.load * 100:.1f}%"
                gpu_memory = f"{gpu.memoryUsed}/{gpu.memoryTotal} MB"
            else:
                gpu_load = "Not available"
                gpu_memory = "Not available"

                self.logger.info(
                    f"Resource usage - CPU: {cpu_percent}%, RAM: {ram_percent}%, GPU Load: {gpu_load}, GPU Memory: {gpu_memory}"
                )

    def stop(self):
        self.stopped.set()
