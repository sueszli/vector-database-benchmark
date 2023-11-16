import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(conf_mat, hide_spines=False, hide_ticks=False, figsize=None, cmap=None, colorbar=False, show_absolute=True, show_normed=False, norm_colormap=None, class_names=None, figure=None, axis=None, fontcolor_threshold=0.5):
    if False:
        for i in range(10):
            print('nop')
    'Plot a confusion matrix via matplotlib.\n\n    Parameters\n    -----------\n    conf_mat : array-like, shape = [n_classes, n_classes]\n        Confusion matrix from evaluate.confusion matrix.\n\n    hide_spines : bool (default: False)\n        Hides axis spines if True.\n\n    hide_ticks : bool (default: False)\n        Hides axis ticks if True\n\n    figsize : tuple (default: (2.5, 2.5))\n        Height and width of the figure\n\n    cmap : matplotlib colormap (default: `None`)\n        Uses matplotlib.pyplot.cm.Blues if `None`\n\n    colorbar : bool (default: False)\n        Shows a colorbar if True\n\n    show_absolute : bool (default: True)\n        Shows absolute confusion matrix coefficients if True.\n        At least one of  `show_absolute` or `show_normed`\n        must be True.\n\n    show_normed : bool (default: False)\n        Shows normed confusion matrix coefficients if True.\n        The normed confusion matrix coefficients give the\n        proportion of training examples per class that are\n        assigned the correct label.\n        At least one of  `show_absolute` or `show_normed`\n        must be True.\n\n    norm_colormap : bool (default: False)\n        Matplotlib color normalization object to normalize the\n        color scale, e.g., `matplotlib.colors.LogNorm()`.\n\n    class_names : array-like, shape = [n_classes] (default: None)\n        List of class names.\n        If not `None`, ticks will be set to these values.\n\n    figure : None or Matplotlib figure  (default: None)\n        If None will create a new figure.\n\n    axis : None or Matplotlib figure axis (default: None)\n        If None will create a new axis.\n\n    fontcolor_threshold : Float (default: 0.5)\n        Sets a threshold for choosing black and white font colors\n        for the cells. By default all values larger than 0.5 times\n        the maximum cell value are converted to white, and everything\n        equal or smaller than 0.5 times the maximum cell value are converted\n        to black.\n\n    Returns\n    -----------\n    fig, ax : matplotlib.pyplot subplot objects\n        Figure and axis elements of the subplot.\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/\n\n    '
    if not (show_absolute or show_normed):
        raise AssertionError('Both show_absolute and show_normed are False')
    if class_names is not None and len(class_names) != len(conf_mat):
        raise AssertionError('len(class_names) should be equal to number ofclasses in the dataset')
    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype('float') / total_samples
    if figure is None and axis is None:
        (fig, ax) = plt.subplots(figsize=figsize)
    elif axis is None:
        fig = figure
        ax = fig.add_subplot(1, 1, 1)
    else:
        (fig, ax) = (figure, axis)
    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues
    if figsize is None:
        figsize = (len(conf_mat) * 1.25, len(conf_mat) * 1.25)
    if show_normed:
        matshow = ax.matshow(normed_conf_mat, cmap=cmap, norm=norm_colormap)
    else:
        matshow = ax.matshow(conf_mat, cmap=cmap, norm=norm_colormap)
    if colorbar:
        fig.colorbar(matshow)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ''
            if show_absolute:
                num = conf_mat[i, j].astype(np.int64)
                cell_text += format(num, 'd')
                if show_normed:
                    cell_text += '\n' + '('
                    cell_text += format(normed_conf_mat[i, j], '.2f') + ')'
            else:
                cell_text += format(normed_conf_mat[i, j], '.2f')
            if show_normed:
                ax.text(x=j, y=i, s=cell_text, va='center', ha='center', color='white' if normed_conf_mat[i, j] > 1 * fontcolor_threshold else 'black')
            else:
                ax.text(x=j, y=i, s=cell_text, va='center', ha='center', color='white' if conf_mat[i, j] > np.max(conf_mat) * fontcolor_threshold else 'black')
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right', rotation_mode='anchor')
        plt.yticks(tick_marks, class_names)
    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    return (fig, ax)