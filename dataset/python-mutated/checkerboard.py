import numpy as np
from matplotlib.pyplot import subplots
from matplotlib.table import Table

def checkerboard_plot(ary, cell_colors=('white', 'black'), font_colors=('black', 'white'), fmt='%.1f', figsize=None, row_labels=None, col_labels=None, fontsize=None):
    if False:
        return 10
    "\n    Plot a checkerboard table / heatmap via matplotlib.\n\n    Parameters\n    -----------\n    ary : array-like, shape = [n, m]\n        A 2D Nnumpy array.\n    cell_colors : tuple or list (default: ('white', 'black'))\n        Tuple or list containing the two colors of the\n        checkerboard pattern.\n    font_colors : tuple or list (default: ('black', 'white'))\n        Font colors corresponding to the cell colors.\n    figsize : tuple (default: (2.5, 2.5))\n        Height and width of the figure\n    fmt : str (default: '%.1f')\n        Python string formatter for cell values.\n        The default '%.1f' results in floats with 1 digit after\n        the decimal point. Use '%d' to show numbers as integers.\n    row_labels : list (default: None)\n        List of the row labels. Uses the array row\n        indices 0 to n by default.\n    col_labels : list (default: None)\n        List of the column labels. Uses the array column\n        indices 0 to m by default.\n    fontsize : int (default: None)\n        Specifies the font size of the checkerboard table.\n        Uses matplotlib's default if None.\n\n    Returns\n    -----------\n    fig : matplotlib Figure object.\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/plotting/checkerboard_plot/\n\n    "
    (fig, ax) = subplots(figsize=figsize)
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    (n_rows, n_cols) = ary.shape
    if row_labels is None:
        row_labels = np.arange(n_rows)
    if col_labels is None:
        col_labels = np.arange(n_cols)
    (width, height) = (1.0 / n_cols, 1.0 / n_rows)
    for ((row_idx, col_idx), cell_val) in np.ndenumerate(ary):
        idx = (col_idx + row_idx) % 2
        tb.add_cell(row_idx, col_idx, width, height, text=fmt % cell_val, loc='center', facecolor=cell_colors[idx])
    for (row_idx, label) in enumerate(row_labels):
        tb.add_cell(row_idx, -1, width, height, text=label, loc='right', edgecolor='none', facecolor='none')
    for (col_idx, label) in enumerate(col_labels):
        tb.add_cell(-1, col_idx, width, height / 2.0, text=label, loc='center', edgecolor='none', facecolor='none')
    for ((row_idx, col_idx), cell_val) in np.ndenumerate(ary):
        idx = (col_idx + row_idx) % 2
        tb._cells[row_idx, col_idx]._text.set_color(font_colors[idx])
    ax.add_table(tb)
    tb.set_fontsize(fontsize)
    return fig