import matplotlib.pyplot as plt
import numpy as np

def scatterplotmatrix(X, fig_axes=None, names=None, figsize=(8, 8), alpha=1.0, **kwargs):
    if False:
        return 10
    '\n    Lower triangular of a scatterplot matrix\n\n    Parameters\n    -----------\n    X : array-like, shape={num_examples, num_features}\n      Design matrix containing data instances (examples)\n      with multiple exploratory variables (features).\n\n    fix_axes : tuple (default: None)\n      A `(fig, axes)` tuple, where fig is an figure object\n      and axes is an axes object created via matplotlib,\n      for example, by calling the pyplot `subplot` function\n      `fig, axes = plt.subplots(...)`\n\n    names : list (default: None)\n      A list of string names, which should have the same number\n      of elements as there are features (columns) in `X`.\n\n    figsize : tuple (default: (8, 8))\n      Height and width of the subplot grid. Ignored if\n      fig_axes is not `None`.\n\n    alpha : float (default: 1.0)\n      Transparency for both the scatter plots and the\n      histograms along the diagonal.\n\n    **kwargs : kwargs\n      Keyword arguments for the scatterplots.\n\n    Returns\n    --------\n    fix_axes : tuple\n      A `(fig, axes)` tuple, where fig is an figure object\n      and axes is an axes object created via matplotlib,\n      for example, by calling the pyplot `subplot` function\n      `fig, axes = plt.subplots(...)`\n\n    Examples\n    ----------\n    For more usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/plotting/scatterplotmatrix/\n\n    '
    (num_examples, num_features) = X.shape
    if fig_axes is None:
        (fig, axes) = plt.subplots(nrows=num_features, ncols=num_features, figsize=figsize)
    else:
        (fig, axes) = fig_axes
    if names is None:
        names = ['X%d' % (i + 1) for i in range(num_features)]
    for (i, j) in zip(*np.triu_indices_from(axes, k=1)):
        axes[j, i].scatter(X[:, j], X[:, i], alpha=alpha, **kwargs)
        axes[j, i].set_xlabel(names[j])
        axes[j, i].set_ylabel(names[i])
        axes[i, j].set_axis_off()
    for i in range(num_features):
        axes[i, i].hist(X[:, i], alpha=alpha)
        axes[i, i].set_ylabel('Count')
        axes[i, i].set_xlabel(names[i])
    return (fig, axes)