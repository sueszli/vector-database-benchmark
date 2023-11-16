from __future__ import annotations
from typing import Callable
from optuna._experimental import experimental_func
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization._rank import _get_rank_info
from optuna.visualization._rank import _get_tick_info
from optuna.visualization._rank import _RankPlotInfo
from optuna.visualization._rank import _RankSubplotInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports
if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import PathCollection
    from optuna.visualization.matplotlib._matplotlib_imports import plt
_logger = get_logger(__name__)

@experimental_func('3.2.0')
def plot_rank(study: Study, params: list[str] | None=None, *, target: Callable[[FrozenTrial], float] | None=None, target_name: str='Objective Value') -> 'Axes':
    if False:
        while True:
            i = 10
    'Plot parameter relations as scatter plots with colors indicating ranks of target value.\n\n    Note that trials missing the specified parameters will not be plotted.\n\n    .. seealso::\n        Please refer to :func:`optuna.visualization.plot_rank` for an example.\n\n    Warnings:\n        Output figures of this Matplotlib-based\n        :func:`~optuna.visualization.matplotlib.plot_rank` function would be different from\n        those of the Plotly-based :func:`~optuna.visualization.plot_rank`.\n\n    Example:\n\n        The following code snippet shows how to plot the parameter relationship as a rank plot.\n\n        .. plot::\n\n            import optuna\n\n\n            def objective(trial):\n                x = trial.suggest_float("x", -100, 100)\n                y = trial.suggest_categorical("y", [-1, 0, 1])\n\n                c0 = 400 - (x + y)**2\n                trial.set_user_attr("constraint", [c0])\n\n                return x ** 2 + y\n\n\n            def constraints(trial):\n                return trial.user_attrs["constraint"]\n\n\n            sampler = optuna.samplers.TPESampler(seed=10, constraints_func=constraints)\n            study = optuna.create_study(sampler=sampler)\n            study.optimize(objective, n_trials=30)\n\n            optuna.visualization.matplotlib.plot_rank(study, params=["x", "y"])\n\n    Args:\n        study:\n            A :class:`~optuna.study.Study` object whose trials are plotted for their target values.\n        params:\n            Parameter list to visualize. The default is all parameters.\n        target:\n            A function to specify the value to display. If it is :obj:`None` and ``study`` is being\n            used for single-objective optimization, the objective values are plotted.\n\n            .. note::\n                Specify this argument if ``study`` is being used for multi-objective optimization.\n        target_name:\n            Target\'s name to display on the color bar.\n\n    Returns:\n        A :class:`matplotlib.axes.Axes` object.\n    '
    _imports.check()
    _logger.warning('Output figures of this Matplotlib-based `plot_rank` function would be different from those of the Plotly-based `plot_rank`.')
    info = _get_rank_info(study, params, target, target_name)
    return _get_rank_plot(info)

def _get_rank_plot(info: _RankPlotInfo) -> 'Axes':
    if False:
        while True:
            i = 10
    params = info.params
    sub_plot_infos = info.sub_plot_infos
    plt.style.use('ggplot')
    title = f'Rank ({info.target_name})'
    n_params = len(params)
    if n_params == 0:
        (_, ax) = plt.subplots()
        ax.set_title(title)
        return ax
    if n_params == 1 or n_params == 2:
        (fig, axs) = plt.subplots()
        axs.set_title(title)
        pc = _add_rank_subplot(axs, sub_plot_infos[0][0])
    else:
        (fig, axs) = plt.subplots(n_params, n_params)
        fig.suptitle(title)
        for x_i in range(n_params):
            for y_i in range(n_params):
                ax = axs[x_i, y_i]
                pc = _add_rank_subplot(ax, sub_plot_infos[x_i][y_i], set_x_label=x_i == n_params - 1, set_y_label=y_i == 0)
    tick_info = _get_tick_info(info.zs)
    pc.set_cmap(plt.get_cmap('RdYlBu_r'))
    cbar = fig.colorbar(pc, ax=axs, ticks=tick_info.coloridxs)
    cbar.ax.set_yticklabels(tick_info.text)
    cbar.outline.set_edgecolor('gray')
    return axs

def _add_rank_subplot(ax: 'Axes', info: _RankSubplotInfo, set_x_label: bool=True, set_y_label: bool=True) -> 'PathCollection':
    if False:
        for i in range(10):
            print('nop')
    if set_x_label:
        ax.set_xlabel(info.xaxis.name)
    if set_y_label:
        ax.set_ylabel(info.yaxis.name)
    if not info.xaxis.is_cat:
        ax.set_xlim(info.xaxis.range[0], info.xaxis.range[1])
    if not info.yaxis.is_cat:
        ax.set_ylim(info.yaxis.range[0], info.yaxis.range[1])
    if info.xaxis.is_log:
        ax.set_xscale('log')
    if info.yaxis.is_log:
        ax.set_yscale('log')
    return ax.scatter(x=info.xs, y=info.ys, c=info.colors / 255, edgecolors='grey')