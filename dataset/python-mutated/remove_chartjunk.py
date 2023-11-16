def remove_borders(axes, left=False, bottom=False, right=True, top=True):
    if False:
        while True:
            i = 10
    'Remove chart junk from matplotlib plots.\n\n    Parameters\n    ----------\n    axes : iterable\n        An iterable containing plt.gca()\n        or plt.subplot() objects, e.g. [plt.gca()].\n    left : bool (default: `False`)\n        Hide left axis spine if True.\n    bottom : bool (default: `False`)\n        Hide bottom axis spine if True.\n    right : bool (default: `True`)\n        Hide right axis spine if True.\n    top : bool (default: `True`)\n        Hide top axis spine if True.\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/plotting/remove_chartjunk/\n\n    '
    for ax in axes:
        ax.spines['top'].set_visible(not top)
        ax.spines['right'].set_visible(not right)
        ax.spines['bottom'].set_visible(not bottom)
        ax.spines['left'].set_visible(not left)
        if bottom:
            ax.tick_params(bottom='off', labelbottom='off')
        if top:
            ax.tick_params(top='off')
        if left:
            ax.tick_params(left='off', labelleft='off')
        if right:
            ax.tick_params(right='off')