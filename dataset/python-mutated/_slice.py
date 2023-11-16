from __future__ import annotations
from collections import defaultdict
import math
from typing import Any
from typing import Callable
from optuna._experimental import experimental_func
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization._slice import _get_slice_plot_info
from optuna.visualization._slice import _PlotValues
from optuna.visualization._slice import _SlicePlotInfo
from optuna.visualization._slice import _SliceSubplotInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports
if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import Colormap
    from optuna.visualization.matplotlib._matplotlib_imports import matplotlib
    from optuna.visualization.matplotlib._matplotlib_imports import PathCollection
    from optuna.visualization.matplotlib._matplotlib_imports import plt

@experimental_func('2.2.0')
def plot_slice(study: Study, params: list[str] | None=None, *, target: Callable[[FrozenTrial], float] | None=None, target_name: str='Objective Value') -> 'Axes':
    if False:
        return 10
    'Plot the parameter relationship as slice plot in a study with Matplotlib.\n\n    .. seealso::\n        Please refer to :func:`optuna.visualization.plot_slice` for an example.\n\n    Example:\n\n        The following code snippet shows how to plot the parameter relationship as slice plot.\n\n        .. plot::\n\n            import optuna\n\n\n            def objective(trial):\n                x = trial.suggest_float("x", -100, 100)\n                y = trial.suggest_categorical("y", [-1, 0, 1])\n                return x ** 2 + y\n\n\n            sampler = optuna.samplers.TPESampler(seed=10)\n            study = optuna.create_study(sampler=sampler)\n            study.optimize(objective, n_trials=10)\n\n            optuna.visualization.matplotlib.plot_slice(study, params=["x", "y"])\n\n    Args:\n        study:\n            A :class:`~optuna.study.Study` object whose trials are plotted for their target values.\n        params:\n            Parameter list to visualize. The default is all parameters.\n        target:\n            A function to specify the value to display. If it is :obj:`None` and ``study`` is being\n            used for single-objective optimization, the objective values are plotted.\n\n            .. note::\n                Specify this argument if ``study`` is being used for multi-objective optimization.\n        target_name:\n            Target\'s name to display on the axis label.\n\n\n    Returns:\n        A :class:`matplotlib.axes.Axes` object.\n    '
    _imports.check()
    return _get_slice_plot(_get_slice_plot_info(study, params, target, target_name))

def _get_slice_plot(info: _SlicePlotInfo) -> 'Axes':
    if False:
        while True:
            i = 10
    if len(info.subplots) == 0:
        (_, ax) = plt.subplots()
        return ax
    cmap = plt.get_cmap('Blues')
    padding_ratio = 0.05
    plt.style.use('ggplot')
    if len(info.subplots) == 1:
        (fig, axs) = plt.subplots()
        axs.set_title('Slice Plot')
        sc = _generate_slice_subplot(info.subplots[0], axs, cmap, padding_ratio, info.target_name)
    else:
        min_figwidth = matplotlib.rcParams['figure.figsize'][0] / 2
        fighight = matplotlib.rcParams['figure.figsize'][1]
        (fig, axs) = plt.subplots(1, len(info.subplots), sharey=True, figsize=(min_figwidth * len(info.subplots), fighight))
        fig.suptitle('Slice Plot')
        for (i, subplot) in enumerate(info.subplots):
            ax = axs[i]
            sc = _generate_slice_subplot(subplot, ax, cmap, padding_ratio, info.target_name)
    axcb = fig.colorbar(sc, ax=axs)
    axcb.set_label('Trial')
    return axs

def _generate_slice_subplot(subplot_info: _SliceSubplotInfo, ax: 'Axes', cmap: 'Colormap', padding_ratio: float, target_name: str) -> 'PathCollection':
    if False:
        while True:
            i = 10
    ax.set(xlabel=subplot_info.param_name, ylabel=target_name)
    scale = None
    feasible = _PlotValues([], [], [])
    infeasible = _PlotValues([], [], [])
    for (x, y, num, c) in zip(subplot_info.x, subplot_info.y, subplot_info.trial_numbers, subplot_info.constraints):
        if x is not None or x != 'None' or y is not None or (y != 'None'):
            if c:
                feasible.x.append(x)
                feasible.y.append(y)
                feasible.trial_numbers.append(num)
            else:
                infeasible.x.append(x)
                infeasible.y.append(y)
                infeasible.trial_numbers.append(num)
    if subplot_info.is_log:
        ax.set_xscale('log')
        scale = 'log'
    if subplot_info.is_numerical:
        feasible_x = feasible.x
        feasible_y = feasible.y
        feasible_c = feasible.trial_numbers
        infeasible_x = infeasible.x
        infeasible_y = infeasible.y
    else:
        (feasible_x, feasible_y, feasible_c) = _get_categorical_plot_values(subplot_info, feasible)
        (infeasible_x, infeasible_y, _) = _get_categorical_plot_values(subplot_info, infeasible)
        scale = 'categorical'
    xlim = _calc_lim_with_padding(feasible_x + infeasible_x, padding_ratio, scale)
    ax.set_xlim(xlim[0], xlim[1])
    sc = ax.scatter(feasible_x, feasible_y, c=feasible_c, cmap=cmap, edgecolors='grey')
    ax.scatter(infeasible_x, infeasible_y, c='#cccccc', label='Infeasible Trial')
    ax.label_outer()
    return sc

def _get_categorical_plot_values(subplot_info: _SliceSubplotInfo, values: _PlotValues) -> tuple[list[Any], list[float], list[int]]:
    if False:
        i = 10
        return i + 15
    assert subplot_info.x_labels is not None
    value_x = []
    value_y = []
    value_c = []
    points_dict = defaultdict(list)
    for (x, y, number) in zip(values.x, values.y, values.trial_numbers):
        points_dict[x].append((y, number))
    for x_label in subplot_info.x_labels:
        for (y, number) in points_dict[x_label]:
            value_x.append(str(x_label))
            value_y.append(y)
            value_c.append(number)
    return (value_x, value_y, value_c)

def _calc_lim_with_padding(values: list[Any], padding_ratio: float, scale: str | None) -> tuple[float, float]:
    if False:
        print('Hello World!')
    value_max = max(values)
    value_min = min(values)
    if scale == 'log':
        padding = (math.log10(value_max) - math.log10(value_min)) * padding_ratio
        return (math.pow(10, math.log10(value_min) - padding), math.pow(10, math.log10(value_max) + padding))
    elif scale == 'categorical':
        width = len(set(values)) - 1
        padding = width * padding_ratio
        return (-padding, width + padding)
    else:
        padding = (value_max - value_min) * padding_ratio
        return (value_min - padding, value_max + padding)