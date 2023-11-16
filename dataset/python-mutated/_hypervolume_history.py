from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from optuna._experimental import experimental_func
from optuna.study import Study
from optuna.visualization._hypervolume_history import _get_hypervolume_history_info
from optuna.visualization._hypervolume_history import _HypervolumeHistoryInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports
if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt

@experimental_func('3.3.0')
def plot_hypervolume_history(study: Study, reference_point: Sequence[float]) -> 'Axes':
    if False:
        i = 10
        return i + 15
    'Plot hypervolume history of all trials in a study with Matplotlib.\n\n    Example:\n\n        The following code snippet shows how to plot optimization history.\n\n        .. plot::\n\n            import optuna\n            import matplotlib.pyplot as plt\n\n\n            def objective(trial):\n                x = trial.suggest_float("x", 0, 5)\n                y = trial.suggest_float("y", 0, 3)\n\n                v0 = 4 * x ** 2 + 4 * y ** 2\n                v1 = (x - 5) ** 2 + (y - 5) ** 2\n                return v0, v1\n\n\n            study = optuna.create_study(directions=["minimize", "minimize"])\n            study.optimize(objective, n_trials=50)\n\n            reference_point=[100, 50]\n            optuna.visualization.matplotlib.plot_hypervolume_history(study, reference_point)\n            plt.tight_layout()\n\n        .. note::\n            You need to adjust the size of the plot by yourself using ``plt.tight_layout()`` or\n            ``plt.savefig(IMAGE_NAME, bbox_inches=\'tight\')``.\n\n    Args:\n        study:\n            A :class:`~optuna.study.Study` object whose trials are plotted for their hypervolumes.\n            The number of objectives must be 2 or more.\n\n        reference_point:\n            A reference point to use for hypervolume computation.\n            The dimension of the reference point must be the same as the number of objectives.\n\n    Returns:\n        A :class:`matplotlib.axes.Axes` object.\n    '
    _imports.check()
    if not study._is_multi_objective():
        raise ValueError('Study must be multi-objective. For single-objective optimization, please use plot_optimization_history instead.')
    if len(reference_point) != len(study.directions):
        raise ValueError('The dimension of the reference point must be the same as the number of objectives.')
    info = _get_hypervolume_history_info(study, np.asarray(reference_point, dtype=np.float64))
    return _get_hypervolume_history_plot(info)

def _get_hypervolume_history_plot(info: _HypervolumeHistoryInfo) -> 'Axes':
    if False:
        print('Hello World!')
    plt.style.use('ggplot')
    (_, ax) = plt.subplots()
    ax.set_title('Hypervolume History Plot')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Hypervolume')
    cmap = plt.get_cmap('tab10')
    ax.plot(info.trial_numbers, info.values, marker='o', color=cmap(0), alpha=0.5)
    return ax