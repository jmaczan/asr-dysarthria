import threading
import time


class StatusUpdater(threading.Thread):
    def __init__(self, study, logger, interval=3600):
        threading.Thread.__init__(self)
        self.study = study
        self.logger = logger
        self.interval = interval
        self.stopped = threading.Event()

    def run(self):
        while not self.stopped.wait(self.interval):
            best_trial = self.study.best_trial
            self.logger.info(f"Current best value: {best_trial.value}")
            self.logger.info(f"Current best params: {best_trial.params}")

    def stop(self):
        self.stopped.set()
