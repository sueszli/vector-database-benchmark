from optuna._experimental import experimental_func
from optuna.logging import get_logger
from optuna.study import Study
from optuna.visualization._intermediate_values import _get_intermediate_plot_info
from optuna.visualization._intermediate_values import _IntermediatePlotInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports
if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt
_logger = get_logger(__name__)

@experimental_func('2.2.0')
def plot_intermediate_values(study: Study) -> 'Axes':
    if False:
        for i in range(10):
            print('nop')
    'Plot intermediate values of all trials in a study with Matplotlib.\n\n    .. note::\n        Please refer to `matplotlib.pyplot.legend\n        <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html>`_\n        to adjust the style of the generated legend.\n\n    Example:\n\n        The following code snippet shows how to plot intermediate values.\n\n        .. plot::\n\n            import optuna\n\n\n            def f(x):\n                return (x - 2) ** 2\n\n\n            def df(x):\n                return 2 * x - 4\n\n\n            def objective(trial):\n                lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)\n\n                x = 3\n                for step in range(128):\n                    y = f(x)\n\n                    trial.report(y, step=step)\n                    if trial.should_prune():\n                        raise optuna.TrialPruned()\n\n                    gy = df(x)\n                    x -= gy * lr\n\n                return y\n\n\n            sampler = optuna.samplers.TPESampler(seed=10)\n            study = optuna.create_study(sampler=sampler)\n            study.optimize(objective, n_trials=16)\n\n            optuna.visualization.matplotlib.plot_intermediate_values(study)\n\n    .. seealso::\n        Please refer to :func:`optuna.visualization.plot_intermediate_values` for an example.\n\n    Args:\n        study:\n            A :class:`~optuna.study.Study` object whose trials are plotted for their intermediate\n            values.\n\n    Returns:\n        A :class:`matplotlib.axes.Axes` object.\n    '
    _imports.check()
    return _get_intermediate_plot(_get_intermediate_plot_info(study))

def _get_intermediate_plot(info: _IntermediatePlotInfo) -> 'Axes':
    if False:
        return 10
    plt.style.use('ggplot')
    (_, ax) = plt.subplots(tight_layout=True)
    ax.set_title('Intermediate Values Plot')
    ax.set_xlabel('Step')
    ax.set_ylabel('Intermediate Value')
    cmap = plt.get_cmap('tab20')
    trial_infos = info.trial_infos
    for (i, tinfo) in enumerate(trial_infos):
        ax.plot(tuple((x for (x, _) in tinfo.sorted_intermediate_values)), tuple((y for (_, y) in tinfo.sorted_intermediate_values)), color=cmap(i) if tinfo.feasible else '#CCCCCC', marker='.', alpha=0.7, label='Trial{}'.format(tinfo.trial_number))
    if len(trial_infos) >= 2:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
    return ax