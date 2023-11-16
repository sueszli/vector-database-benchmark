from typing import Callable, List, Optional, Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3.common.monitor import load_results
X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100

def rolling_window(array: np.ndarray, window: int) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    '\n    Apply a rolling window to a np.ndarray\n\n    :param array: the input Array\n    :param window: length of the rolling window\n    :return: rolling window on the input array\n    '
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = (*array.strides, array.strides[-1])
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

def window_func(var_1: np.ndarray, var_2: np.ndarray, window: int, func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        print('Hello World!')
    '\n    Apply a function to the rolling window of 2 arrays\n\n    :param var_1: variable 1\n    :param var_2: variable 2\n    :param window: length of the rolling window\n    :param func: function to apply on the rolling window on variable 2 (such as np.mean)\n    :return:  the rolling output with applied function\n    '
    var_2_window = rolling_window(var_2, window)
    function_on_var2 = func(var_2_window, axis=-1)
    return (var_1[window - 1:], function_on_var2)

def ts2xy(data_frame: pd.DataFrame, x_axis: str) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Decompose a data frame variable to x ans ys\n\n    :param data_frame: the input data\n    :param x_axis: the axis for the x and y output\n        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')\n    :return: the x and y output\n    "
    if x_axis == X_TIMESTEPS:
        x_var = np.cumsum(data_frame.l.values)
        y_var = data_frame.r.values
    elif x_axis == X_EPISODES:
        x_var = np.arange(len(data_frame))
        y_var = data_frame.r.values
    elif x_axis == X_WALLTIME:
        x_var = data_frame.t.values / 3600.0
        y_var = data_frame.r.values
    else:
        raise NotImplementedError
    return (x_var, y_var)

def plot_curves(xy_list: List[Tuple[np.ndarray, np.ndarray]], x_axis: str, title: str, figsize: Tuple[int, int]=(8, 2)) -> None:
    if False:
        return 10
    "\n    plot the curves\n\n    :param xy_list: the x and y coordinates to plot\n    :param x_axis: the axis for the x and y output\n        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')\n    :param title: the title of the plot\n    :param figsize: Size of the figure (width, height)\n    "
    plt.figure(title, figsize=figsize)
    max_x = max((xy[0][-1] for xy in xy_list))
    min_x = 0
    for (_, (x, y)) in enumerate(xy_list):
        plt.scatter(x, y, s=2)
        if x.shape[0] >= EPISODES_WINDOW:
            (x, y_mean) = window_func(x, y, EPISODES_WINDOW, np.mean)
            plt.plot(x, y_mean)
    plt.xlim(min_x, max_x)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel('Episode Rewards')
    plt.tight_layout()

def plot_results(dirs: List[str], num_timesteps: Optional[int], x_axis: str, task_name: str, figsize: Tuple[int, int]=(8, 2)) -> None:
    if False:
        for i in range(10):
            print('nop')
    "\n    Plot the results using csv files from ``Monitor`` wrapper.\n\n    :param dirs: the save location of the results to plot\n    :param num_timesteps: only plot the points below this value\n    :param x_axis: the axis for the x and y output\n        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')\n    :param task_name: the title of the task to plot\n    :param figsize: Size of the figure (width, height)\n    "
    data_frames = []
    for folder in dirs:
        data_frame = load_results(folder)
        if num_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        data_frames.append(data_frame)
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
    plot_curves(xy_list, x_axis, task_name, figsize)