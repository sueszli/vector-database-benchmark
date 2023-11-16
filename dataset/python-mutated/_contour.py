from __future__ import annotations
from typing import Callable
from typing import Sequence
import numpy as np
from optuna._experimental import experimental_func
from optuna._imports import try_import
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization._contour import _AxisInfo
from optuna.visualization._contour import _ContourInfo
from optuna.visualization._contour import _get_contour_info
from optuna.visualization._contour import _PlotValues
from optuna.visualization._contour import _SubContourInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports
with try_import() as _optuna_imports:
    import scipy
if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import Colormap
    from optuna.visualization.matplotlib._matplotlib_imports import ContourSet
    from optuna.visualization.matplotlib._matplotlib_imports import plt
_logger = get_logger(__name__)
CONTOUR_POINT_NUM = 100

@experimental_func('2.2.0')
def plot_contour(study: Study, params: list[str] | None=None, *, target: Callable[[FrozenTrial], float] | None=None, target_name: str='Objective Value') -> 'Axes':
    if False:
        for i in range(10):
            print('nop')
    'Plot the parameter relationship as contour plot in a study with Matplotlib.\n\n    Note that, if a parameter contains missing values, a trial with missing values is not plotted.\n\n    .. seealso::\n        Please refer to :func:`optuna.visualization.plot_contour` for an example.\n\n    Warnings:\n        Output figures of this Matplotlib-based\n        :func:`~optuna.visualization.matplotlib.plot_contour` function would be different from\n        those of the Plotly-based :func:`~optuna.visualization.plot_contour`.\n\n    Example:\n\n        The following code snippet shows how to plot the parameter relationship as contour plot.\n\n        .. plot::\n\n            import optuna\n\n\n            def objective(trial):\n                x = trial.suggest_float("x", -100, 100)\n                y = trial.suggest_categorical("y", [-1, 0, 1])\n                return x ** 2 + y\n\n\n            sampler = optuna.samplers.TPESampler(seed=10)\n            study = optuna.create_study(sampler=sampler)\n            study.optimize(objective, n_trials=30)\n\n            optuna.visualization.matplotlib.plot_contour(study, params=["x", "y"])\n\n    Args:\n        study:\n            A :class:`~optuna.study.Study` object whose trials are plotted for their target values.\n        params:\n            Parameter list to visualize. The default is all parameters.\n        target:\n            A function to specify the value to display. If it is :obj:`None` and ``study`` is being\n            used for single-objective optimization, the objective values are plotted.\n\n            .. note::\n                Specify this argument if ``study`` is being used for multi-objective optimization.\n        target_name:\n            Target\'s name to display on the color bar.\n\n    Returns:\n        A :class:`matplotlib.axes.Axes` object.\n\n    .. note::\n        The colormap is reversed when the ``target`` argument isn\'t :obj:`None` or ``direction``\n        of :class:`~optuna.study.Study` is ``minimize``.\n    '
    _imports.check()
    _logger.warning('Output figures of this Matplotlib-based `plot_contour` function would be different from those of the Plotly-based `plot_contour`.')
    info = _get_contour_info(study, params, target, target_name)
    return _get_contour_plot(info)

def _get_contour_plot(info: _ContourInfo) -> 'Axes':
    if False:
        print('Hello World!')
    sorted_params = info.sorted_params
    sub_plot_infos = info.sub_plot_infos
    reverse_scale = info.reverse_scale
    target_name = info.target_name
    if len(sorted_params) <= 1:
        (_, ax) = plt.subplots()
        return ax
    n_params = len(sorted_params)
    plt.style.use('ggplot')
    if n_params == 2:
        (fig, axs) = plt.subplots()
        axs.set_title('Contour Plot')
        cmap = _set_cmap(reverse_scale)
        cs = _generate_contour_subplot(sub_plot_infos[0][0], axs, cmap)
        if isinstance(cs, ContourSet):
            axcb = fig.colorbar(cs)
            axcb.set_label(target_name)
    else:
        (fig, axs) = plt.subplots(n_params, n_params)
        fig.suptitle('Contour Plot')
        cmap = _set_cmap(reverse_scale)
        cs_list = []
        for x_i in range(len(sorted_params)):
            for y_i in range(len(sorted_params)):
                ax = axs[y_i, x_i]
                cs = _generate_contour_subplot(sub_plot_infos[y_i][x_i], ax, cmap)
                if isinstance(cs, ContourSet):
                    cs_list.append(cs)
        if cs_list:
            axcb = fig.colorbar(cs_list[0], ax=axs)
            axcb.set_label(target_name)
    return axs

def _set_cmap(reverse_scale: bool) -> 'Colormap':
    if False:
        i = 10
        return i + 15
    cmap = 'Blues_r' if not reverse_scale else 'Blues'
    return plt.get_cmap(cmap)

class _LabelEncoder:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.labels: list[str] = []

    def fit(self, labels: list[str]) -> '_LabelEncoder':
        if False:
            for i in range(10):
                print('nop')
        self.labels = sorted(set(labels))
        return self

    def transform(self, labels: list[str]) -> list[int]:
        if False:
            for i in range(10):
                print('nop')
        return [self.labels.index(label) for label in labels]

    def fit_transform(self, labels: list[str]) -> list[int]:
        if False:
            while True:
                i = 10
        return self.fit(labels).transform(labels)

    def get_labels(self) -> list[str]:
        if False:
            i = 10
            return i + 15
        return self.labels

    def get_indices(self) -> list[int]:
        if False:
            for i in range(10):
                print('nop')
        return list(range(len(self.labels)))

def _calculate_griddata(info: _SubContourInfo) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[str], list[int], list[str], _PlotValues, _PlotValues]:
    if False:
        return 10
    xaxis = info.xaxis
    yaxis = info.yaxis
    z_values_dict = info.z_values
    x_values = []
    y_values = []
    z_values = []
    for (x_value, y_value) in zip(xaxis.values, yaxis.values):
        if x_value is not None and y_value is not None:
            x_values.append(x_value)
            y_values.append(y_value)
            x_i = xaxis.indices.index(x_value)
            y_i = yaxis.indices.index(y_value)
            z_values.append(z_values_dict[x_i, y_i])
    if len(x_values) == 0 or len(y_values) == 0:
        return (np.array([]), np.array([]), np.array([]), [], [], [], [], _PlotValues([], []), _PlotValues([], []))

    def _calculate_axis_data(axis: _AxisInfo, values: Sequence[str | float]) -> tuple[np.ndarray, list[str], list[int], list[int | float]]:
        if False:
            i = 10
            return i + 15
        cat_param_labels: list[str] = []
        cat_param_pos: list[int] = []
        returned_values: Sequence[int | float]
        if axis.is_cat:
            enc = _LabelEncoder()
            returned_values = enc.fit_transform(list(map(str, values)))
            cat_param_labels = enc.get_labels()
            cat_param_pos = enc.get_indices()
        else:
            returned_values = list(map(lambda x: float(x), values))
        if axis.is_log:
            ci = np.logspace(np.log10(axis.range[0]), np.log10(axis.range[1]), CONTOUR_POINT_NUM)
        else:
            ci = np.linspace(axis.range[0], axis.range[1], CONTOUR_POINT_NUM)
        return (ci, cat_param_labels, cat_param_pos, list(returned_values))
    (xi, cat_param_labels_x, cat_param_pos_x, transformed_x_values) = _calculate_axis_data(xaxis, x_values)
    (yi, cat_param_labels_y, cat_param_pos_y, transformed_y_values) = _calculate_axis_data(yaxis, y_values)
    zi: np.ndarray = np.array([])
    if xaxis.name != yaxis.name:
        zmap = _create_zmap(transformed_x_values, transformed_y_values, z_values, xi, yi)
        zi = _interpolate_zmap(zmap, CONTOUR_POINT_NUM)
    feasible = _PlotValues([], [])
    infeasible = _PlotValues([], [])
    for (x_value, y_value, c) in zip(transformed_x_values, transformed_y_values, info.constraints):
        if c:
            feasible.x.append(x_value)
            feasible.y.append(y_value)
        else:
            infeasible.x.append(x_value)
            infeasible.y.append(y_value)
    return (xi, yi, zi, cat_param_pos_x, cat_param_labels_x, cat_param_pos_y, cat_param_labels_y, feasible, infeasible)

def _generate_contour_subplot(info: _SubContourInfo, ax: 'Axes', cmap: 'Colormap') -> 'ContourSet':
    if False:
        return 10
    if len(info.xaxis.indices) < 2 or len(info.yaxis.indices) < 2:
        ax.label_outer()
        return ax
    ax.set(xlabel=info.xaxis.name, ylabel=info.yaxis.name)
    ax.set_xlim(info.xaxis.range[0], info.xaxis.range[1])
    ax.set_ylim(info.yaxis.range[0], info.yaxis.range[1])
    if info.xaxis.name == info.yaxis.name:
        ax.label_outer()
        return ax
    (xi, yi, zi, x_cat_param_pos, x_cat_param_label, y_cat_param_pos, y_cat_param_label, feasible_plot_values, infeasible_plot_values) = _calculate_griddata(info)
    cs = None
    if len(zi) > 0:
        if info.xaxis.is_log:
            ax.set_xscale('log')
        if info.yaxis.is_log:
            ax.set_yscale('log')
        if info.xaxis.name != info.yaxis.name:
            ax.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
            cs = ax.contourf(xi, yi, zi, 15, cmap=cmap.reversed())
            ax.scatter(feasible_plot_values.x, feasible_plot_values.y, marker='o', c='black', s=20, edgecolors='grey', linewidth=2.0)
            ax.scatter(infeasible_plot_values.x, infeasible_plot_values.y, marker='o', c='#cccccc', s=20, edgecolors='#cccccc', linewidth=2.0)
    if info.xaxis.is_cat:
        ax.set_xticks(x_cat_param_pos)
        ax.set_xticklabels(x_cat_param_label)
    if info.yaxis.is_cat:
        ax.set_yticks(y_cat_param_pos)
        ax.set_yticklabels(y_cat_param_label)
    ax.label_outer()
    return cs

def _create_zmap(x_values: list[int | float], y_values: list[int | float], z_values: list[float], xi: np.ndarray, yi: np.ndarray) -> dict[tuple[int, int], float]:
    if False:
        return 10
    zmap = dict()
    for (x, y, z) in zip(x_values, y_values, z_values):
        xindex = int(np.argmin(np.abs(xi - x)))
        yindex = int(np.argmin(np.abs(yi - y)))
        zmap[xindex, yindex] = z
    return zmap

def _interpolate_zmap(zmap: dict[tuple[int, int], float], contour_plot_num: int) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    a_data = []
    a_row = []
    a_col = []
    b = np.zeros(contour_plot_num ** 2)
    for x in range(contour_plot_num):
        for y in range(contour_plot_num):
            grid_index = y * contour_plot_num + x
            if (x, y) in zmap:
                a_data.append(1)
                a_row.append(grid_index)
                a_col.append(grid_index)
                b[grid_index] = zmap[x, y]
            else:
                for (dx, dy) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if 0 <= x + dx < contour_plot_num and 0 <= y + dy < contour_plot_num:
                        a_data.append(1)
                        a_row.append(grid_index)
                        a_col.append(grid_index)
                        a_data.append(-1)
                        a_row.append(grid_index)
                        a_col.append(grid_index + dy * contour_plot_num + dx)
    z = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix((a_data, (a_row, a_col))), b)
    return z.reshape((contour_plot_num, contour_plot_num))