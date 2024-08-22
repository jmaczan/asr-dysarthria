from typing import Any
from optuna.trial import TrialState


class MonitorCallback:
    def __init__(self, logger):
        self.logger = logger

    def __call__(self, study, trial):
        if trial.state == TrialState.COMPLETE:
            self.logger.info(
                f"Trial {trial.number} finished with value: {trial.value} and parameters: {trial.parameters}"
            )
        elif trial.state == TrialState.PRUNED:
            self.logger.info(f"Trial {trial.number} pruned.")
        elif trial.state == TrialState.FAIL:
            self.logger.error(f"Trial {trial.number} failed.")
