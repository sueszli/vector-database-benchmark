from io import BytesIO
from typing import Any
from typing import Callable
from typing import Sequence
import pytest
from optuna.study import create_study
from optuna.testing.objectives import fail_objective
from optuna.trial import FrozenTrial
from optuna.trial import Trial
import optuna.visualization._intermediate_values
from optuna.visualization._intermediate_values import _get_intermediate_plot_info
from optuna.visualization._intermediate_values import _IntermediatePlotInfo
from optuna.visualization._intermediate_values import _TrialInfo
from optuna.visualization._plotly_imports import go
import optuna.visualization.matplotlib._intermediate_values
from optuna.visualization.matplotlib._matplotlib_imports import plt

def test_intermediate_plot_info() -> None:
    if False:
        print('Hello World!')
    study = create_study(direction='minimize')
    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo(trial_infos=[])

    def objective(trial: Trial, report_intermediate_values: bool) -> float:
        if False:
            print('Hello World!')
        if report_intermediate_values:
            trial.report(1.0, step=0)
            trial.report(2.0, step=1)
        return 0.0
    study = create_study()
    study.optimize(lambda t: objective(t, True), n_trials=1)
    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo(trial_infos=[_TrialInfo(trial_number=0, sorted_intermediate_values=[(0, 1.0), (1, 2.0)], feasible=True)])
    study.optimize(lambda t: objective(t, False), n_trials=1)
    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo(trial_infos=[_TrialInfo(trial_number=0, sorted_intermediate_values=[(0, 1.0), (1, 2.0)], feasible=True)])
    study = create_study()
    study.optimize(lambda t: objective(t, False), n_trials=1)
    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo(trial_infos=[])
    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo(trial_infos=[])

    def objective_with_constraints(trial: Trial) -> float:
        if False:
            i = 10
            return i + 15
        trial.set_user_attr('constraint', [trial.number % 2])
        trial.report(1.0, step=0)
        trial.report(2.0, step=1)
        return 0.0

    def constraints(trial: FrozenTrial) -> Sequence[float]:
        if False:
            i = 10
            return i + 15
        return trial.user_attrs['constraint']
    study = create_study(sampler=optuna.samplers.NSGAIIISampler(constraints_func=constraints))
    study.optimize(objective_with_constraints, n_trials=2)
    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo(trial_infos=[_TrialInfo(trial_number=0, sorted_intermediate_values=[(0, 1.0), (1, 2.0)], feasible=True), _TrialInfo(trial_number=1, sorted_intermediate_values=[(0, 1.0), (1, 2.0)], feasible=False)])

@pytest.mark.parametrize('plotter', [optuna.visualization._intermediate_values._get_intermediate_plot, optuna.visualization.matplotlib._intermediate_values._get_intermediate_plot])
@pytest.mark.parametrize('info', [_IntermediatePlotInfo(trial_infos=[]), _IntermediatePlotInfo(trial_infos=[_TrialInfo(trial_number=0, sorted_intermediate_values=[(0, 1.0), (1, 2.0)], feasible=True)]), _IntermediatePlotInfo(trial_infos=[_TrialInfo(trial_number=0, sorted_intermediate_values=[(0, 1.0), (1, 2.0)], feasible=True), _TrialInfo(trial_number=1, sorted_intermediate_values=[(1, 2.0), (0, 1.0)], feasible=False)])])
def test_plot_intermediate_values(plotter: Callable[[_IntermediatePlotInfo], Any], info: _IntermediatePlotInfo) -> None:
    if False:
        i = 10
        return i + 15
    figure = plotter(info)
    if isinstance(figure, go.Figure):
        figure.write_image(BytesIO())
    else:
        plt.savefig(BytesIO())
        plt.close()