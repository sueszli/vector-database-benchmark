import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
cmap = sns.diverging_palette(130, 10, as_cmap=True).reversed()
sns.set_theme()

class HeatMapVisualizer:

    @staticmethod
    def visSeaborn(heatmap_data, mn_val=0.0, mx_val=1000.0, center=500, labels=['prey', 'predator', 'signed episode length', ''], annot=False, linewidth=0, cmap=cmap, xticklabels=5, yticklabels=5, x_axis=None, y_axis=None, cbar=True, show=True, save=False, save_path=None, y_axis_reversed=False):
        if False:
            for i in range(10):
                print('nop')
        mx_val = mx_val
        if isinstance(heatmap_data, list):
            heatmap_data = np.mean(heatmap_data, axis=0)
        num_dia = heatmap_data.shape[0]
        a = np.tril(np.zeros((num_dia, num_dia)) + mx_val, -1).T + np.tril(np.zeros((num_dia, num_dia)) + mn_val) + np.eye(num_dia)
        mse = np.square(np.subtract(a, heatmap_data)).mean()
        dia = np.diag_indices(num_dia)
        dia_sum = sum(heatmap_data[dia])
        off_dia_sum = np.sum(heatmap_data) - dia_sum
        dia_mean = dia_sum / num_dia
        off_dia_mean = off_dia_sum / (num_dia * num_dia - num_dia)
        print('============================================================')
        print(f'Diagonal sum: {dia_sum}, Off-diagonal sum: {off_dia_sum}')
        print(f'Diagonals mean: {dia_mean}, Off-diagonals mean: {off_dia_mean}')
        print(f'R_={(dia_mean - off_dia_mean) / dia_mean}')
        print(f'Sum={np.sum(heatmap_data)}')
        print(f'MSE={mse}')
        print('============================================================')
        ax = None
        if x_axis is not None and y_axis is not None:
            df = pd.DataFrame(heatmap_data, y_axis, columns=x_axis)
            ax = sns.heatmap(df, vmin=mn_val, vmax=mx_val, center=center, cbar_kws={'label': labels[2]}, annot=annot, linewidth=linewidth, cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels, cbar=cbar)
        else:
            ax = sns.heatmap(heatmap_data, vmin=mn_val, vmax=mx_val, center=center, cbar_kws={'label': labels[2]}, annot=annot, linewidth=linewidth, cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels, cbar=cbar)
        ax.set_xlabel(labels[0], fontsize=15)
        ax.set_ylabel(labels[1], fontsize=15)
        ax.set_title(labels[3], fontsize=15)
        ax.figure.axes[-1].yaxis.label.set_size(17)
        if y_axis_reversed:
            ax.invert_yaxis()
        if save:
            if save_path is None:
                print('Saved in the script place with name: test.png')
                plt.savefig('test.png')
            else:
                print(f'Saved in the script place with name: {save_path}')
                plt.savefig(save_path)
        if show:
            plt.show()
        return ax

    @staticmethod
    def visPlotly(heatmap_data, mn_val=0.0, mx_val=1.0, xrange=None, yrange=None, labels=['prey', 'predator', 'win rate', 'win rate (heatmap)'], cmap='YlGnBu'):
        if False:
            return 10
        if isinstance(heatmap_data, list):
            heatmap_data = np.mean(heatmap_data, axis=0)
        xrange_cfg = xrange if xrange is not None else [i for i in range(0, heatmap_data.shape[1])]
        yrange_cfg = yrange if yrange is not None else [i for i in range(0, heatmap_data.shape[0])]
        xaxis_cfg = dict(type='category', categoryorder='array', categoryarray=xrange_cfg) if xrange is not None else dict()
        yaxis_cfg = dict(autorange='reversed', type='category', categoryorder='array', categoryarray=yrange_cfg) if yrange is not None else dict(autorange='reversed')
        fig = go.Figure(data=go.Heatmap(z=heatmap_data, x=xrange_cfg, y=yrange_cfg, type='heatmap', colorscale=cmap.lower()))
        fig.update_layout(xaxis=xaxis_cfg, yaxis=yaxis_cfg, title=labels[3], xaxis_title=labels[0], yaxis_title=labels[1], legend_title=labels[2])
        fig.show()

    @staticmethod
    def save(*args, **kwargs):
        if False:
            while True:
                i = 10
        filename = kwargs.pop('filename', 'default.png')
        dpi = kwargs.pop('dpi', 400)
        filename = filename if filename.endswith('.png') else filename + '.png'
        kwargs['show'] = False
        ax = HeatMapVisualizer.visSeaborn(*args, **kwargs)
        figure = ax.get_figure()
        figure.savefig(filename, dpi=dpi)

def traj_vis(x, y, bins=50):
    if False:
        i = 10
        return i + 15
    cmap_traj = sns.color_palette('coolwarm', as_cmap=True).reversed()
    plt.clf()
    k = 'pred'
    cmap_traj = LinearSegmentedColormap.from_list(name='test', colors=['red', 'white', 'green']).reversed()
    n = len(y[k])
    y[k].extend(y['prey'])
    x[k].extend(x['prey'])
    hist = 0.8
    weights = [i * hist for i in range(n)]
    weights.extend([-i * hist for i in range(len(y[k]) - n)])
    weights = [1 for i in range(n)]
    weights.extend([-1 for i in range(len(y[k]) - n)])
    plt.hist2d(x[k], y[k], bins=bins, weights=weights, cmap=cmap_traj, vmin=-8, vmax=8)
    cb = plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1
    plt.xticks([])
    plt.yticks([])
    plt.show()