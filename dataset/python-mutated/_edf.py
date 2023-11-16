from __future__ import annotations
from collections.abc import Callable
from collections.abc import Sequence
from optuna._experimental import experimental_func
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization._edf import _get_edf_info
from optuna.visualization.matplotlib._matplotlib_imports import _imports
if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt
_logger = get_logger(__name__)

@experimental_func('2.2.0')
def plot_edf(study: Study | Sequence[Study], *, target: Callable[[FrozenTrial], float] | None=None, target_name: str='Objective Value') -> 'Axes':
    if False:
        return 10
    'Plot the objective value EDF (empirical distribution function) of a study with Matplotlib.\n\n    Note that only the complete trials are considered when plotting the EDF.\n\n    .. seealso::\n        Please refer to :func:`optuna.visualization.plot_edf` for an example,\n        where this function can be replaced with it.\n\n    .. note::\n\n        Please refer to `matplotlib.pyplot.legend\n        <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html>`_\n        to adjust the style of the generated legend.\n\n    Example:\n\n        The following code snippet shows how to plot EDF.\n\n        .. plot::\n\n            import math\n\n            import optuna\n\n\n            def ackley(x, y):\n                a = 20 * math.exp(-0.2 * math.sqrt(0.5 * (x ** 2 + y ** 2)))\n                b = math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))\n                return -a - b + math.e + 20\n\n\n            def objective(trial, low, high):\n                x = trial.suggest_float("x", low, high)\n                y = trial.suggest_float("y", low, high)\n                return ackley(x, y)\n\n\n            sampler = optuna.samplers.RandomSampler(seed=10)\n\n            # Widest search space.\n            study0 = optuna.create_study(study_name="x=[0,5), y=[0,5)", sampler=sampler)\n            study0.optimize(lambda t: objective(t, 0, 5), n_trials=500)\n\n            # Narrower search space.\n            study1 = optuna.create_study(study_name="x=[0,4), y=[0,4)", sampler=sampler)\n            study1.optimize(lambda t: objective(t, 0, 4), n_trials=500)\n\n            # Narrowest search space but it doesn\'t include the global optimum point.\n            study2 = optuna.create_study(study_name="x=[1,3), y=[1,3)", sampler=sampler)\n            study2.optimize(lambda t: objective(t, 1, 3), n_trials=500)\n\n            optuna.visualization.matplotlib.plot_edf([study0, study1, study2])\n\n    Args:\n        study:\n            A target :class:`~optuna.study.Study` object.\n            You can pass multiple studies if you want to compare those EDFs.\n        target:\n            A function to specify the value to display. If it is :obj:`None` and ``study`` is being\n            used for single-objective optimization, the objective values are plotted.\n\n            .. note::\n                Specify this argument if ``study`` is being used for multi-objective optimization.\n        target_name:\n            Target\'s name to display on the axis label.\n\n    Returns:\n        A :class:`matplotlib.axes.Axes` object.\n    '
    _imports.check()
    plt.style.use('ggplot')
    (_, ax) = plt.subplots()
    ax.set_title('Empirical Distribution Function Plot')
    ax.set_xlabel(target_name)
    ax.set_ylabel('Cumulative Probability')
    ax.set_ylim(0, 1)
    cmap = plt.get_cmap('tab20')
    info = _get_edf_info(study, target, target_name)
    edf_lines = info.lines
    if len(edf_lines) == 0:
        return ax
    for (i, (study_name, y_values)) in enumerate(edf_lines):
        ax.plot(info.x_values, y_values, color=cmap(i), alpha=0.7, label=study_name)
    if len(edf_lines) >= 2:
        ax.legend()
    return ax