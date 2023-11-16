from optuna import TrialPruned
from optuna.trial import Trial

def fail_objective(_: Trial) -> float:
    if False:
        for i in range(10):
            print('nop')
    raise ValueError()

def pruned_objective(trial: Trial) -> float:
    if False:
        i = 10
        return i + 15
    raise TrialPruned()