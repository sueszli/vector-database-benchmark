from optuna.distributions import FloatDistribution
from optuna.terminator.improvement.gp.botorch import _BoTorchGaussianProcess
from optuna.trial import create_trial

def test_fit_predict() -> None:
    if False:
        print('Hello World!')
    trials = [create_trial(value=1.0, distributions={'bacon': FloatDistribution(-1.0, 1.0), 'egg': FloatDistribution(-1.0, 1.0)}, params={'bacon': 1.0, 'egg': 0.0}), create_trial(value=-1.0, distributions={'bacon': FloatDistribution(-1.0, 1.0), 'egg': FloatDistribution(-1.0, 1.0)}, params={'bacon': 0.0, 'egg': 1.0})]
    gp = _BoTorchGaussianProcess()
    gp.fit(trials)
    gp.predict_mean_std(trials)

def test_fit_predict_single_trial() -> None:
    if False:
        i = 10
        return i + 15
    trials = [create_trial(value=1.0, distributions={'bacon': FloatDistribution(-1.0, 1.0), 'egg': FloatDistribution(-1.0, 1.0)}, params={'bacon': 1.0, 'egg': 0.0})]
    gp = _BoTorchGaussianProcess()
    gp.fit(trials)
    gp.predict_mean_std(trials)

def test_fit_predict_single_param() -> None:
    if False:
        print('Hello World!')
    trials = [create_trial(value=1.0, distributions={'spam': FloatDistribution(-1.0, 1.0)}, params={'spam': 1.0})]
    gp = _BoTorchGaussianProcess()
    gp.fit(trials)
    gp.predict_mean_std(trials)

def test_fit_predict_non_regular_gram_matrix() -> None:
    if False:
        return 10
    trials = [create_trial(value=1.0, distributions={'bacon': FloatDistribution(-1.0, 1.0), 'egg': FloatDistribution(-1.0, 1.0)}, params={'bacon': 1.0, 'egg': 0.0}), create_trial(value=1.0, distributions={'bacon': FloatDistribution(-1.0, 1.0), 'egg': FloatDistribution(-1.0, 1.0)}, params={'bacon': 1.0, 'egg': 0.0})]
    gp = _BoTorchGaussianProcess()
    gp.fit(trials)
    gp.predict_mean_std(trials)