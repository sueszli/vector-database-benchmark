import numpy as np
from sklearn import covariance
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_network_structure(edge_model, embedding, names, labels, ax=[0.0, 0.0, 1.0, 1.0], figsize=None, corr_threshold=0.02, vmin=0, vmax=0.5, lw=1, alpha=None, cmap_scatter=plt.cm.nipy_spectral, cmap_lc=plt.cm.hot_r, edgecolor=plt.cm.nipy_spectral):
    if False:
        i = 10
        return i + 15
    '\n\n    Parameters\n    ----------\n    edge_model: sklearn.covariance.graph_lasso_.GraphicalLassoCV\n                The model specifications to build the graph\n\n    embedding: array-like of shape [n_components, n_instruments]\n               Transformed embedding vectors\n\n    names: array-like of shape [n_samples, 1]\n           Names of each financial instrument\n\n    labels: array-like of shape [n_instruments, 1]\n            numeric identifier of each cluster\n    ax: list of of 4 floats [left, bottom, width, height]\n        Add an axes to the current figure and make it the current axes (plt.axes\n        official docs)\n\n    figsize: (float, float), optional, default: None\n             Width and height in inches\n\n    corr_threshold: float\n                    Minimum correlation value for which to display points\n    vmin: float\n          Minimum value allowed in the normalised range\n\n    vmax: float\n          Maximum value allowed in the normalised range\n\n    lw: float or sequence of float\n\n    alpha: float between 0 and 1\n           Degree of transparency of the plot\n\n    cmap_scatter: plt.cm\n                  colour-mapping for scatter plots\n    cmap_lc: plt.cm\n             colour-mapping for LineCollection\n\n    edgecolor: plt.cm\n               colour of the borders of the box containing each financial instrument\n               name\n\n    Returns\n    A plot representing the correlation network of the financial instruments\n    -------\n\n    '
    if not isinstance(edge_model, covariance.graph_lasso_.GraphicalLassoCV):
        raise TypeError('edge_model must be of class covariance.graph_lasso_.GraphicalLassoCV ')
    if not isinstance(embedding, (np.ndarray, np.generic)):
        raise TypeError('embedding must be of class ndarray.')
    plt.figure(1, facecolor='w', figsize=figsize)
    plt.clf()
    ax = plt.axes(ax)
    plt.axis('off')
    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = np.abs(np.triu(partial_correlations, k=1)) > corr_threshold
    plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels, cmap=cmap_scatter)
    (start_idx, end_idx) = np.where(non_zero)
    segments = [[embedding[:, start], embedding[:, stop]] for (start, stop) in zip(start_idx, end_idx)]
    corr_values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments, zorder=0, cmap=cmap_lc, norm=plt.Normalize(vmin=vmin, vmax=vmax * corr_values.max()))
    lc.set_array(corr_values)
    lc.set_linewidth(lw=lw * corr_values)
    ax.add_collection(lc)
    n_labels = labels.max()
    for (index, (name, label, (x, y))) in enumerate(zip(names, labels, embedding.T)):
        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + 0.002
        else:
            horizontalalignment = 'right'
            x = x - 0.002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + 0.002
        else:
            verticalalignment = 'top'
            y = y - 0.002
        plt.text(x, y, name, size=10, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, bbox=dict(facecolor='w', edgecolor=edgecolor(label / float(n_labels)), alpha=alpha))
        plt.xlim(embedding[0].min() - 0.15 * embedding[0].ptp(), embedding[0].max() + 0.1 * embedding[0].ptp())
        plt.ylim(embedding[1].min() - 0.03 * embedding[1].ptp(), embedding[1].min() + 0.03 * embedding[1].ptp())
        plt.show()