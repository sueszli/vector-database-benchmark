"""Create a mosaic plot from a contingency table.

It allows to visualize multivariate categorical data in a rigorous
and informative way.

see the docstring of the mosaic function for more informations.
"""
from statsmodels.compat.python import lrange, lzip
from itertools import product
import numpy as np
from numpy import array, cumsum, iterable, r_
from pandas import DataFrame
from statsmodels.graphics import utils
__all__ = ['mosaic']

def _normalize_split(proportion):
    if False:
        return 10
    '\n    return a list of proportions of the available space given the division\n    if only a number is given, it will assume a split in two pieces\n    '
    if not iterable(proportion):
        if proportion == 0:
            proportion = array([0.0, 1.0])
        elif proportion >= 1:
            proportion = array([1.0, 0.0])
        elif proportion < 0:
            raise ValueError('proportions should be positive,given value: {}'.format(proportion))
        else:
            proportion = array([proportion, 1.0 - proportion])
    proportion = np.asarray(proportion, dtype=float)
    if np.any(proportion < 0):
        raise ValueError('proportions should be positive,given value: {}'.format(proportion))
    if np.allclose(proportion, 0):
        raise ValueError('at least one proportion should be greater than zero'.format(proportion))
    if len(proportion) < 2:
        return array([0.0, 1.0])
    left = r_[0, cumsum(proportion)]
    left /= left[-1] * 1.0
    return left

def _split_rect(x, y, width, height, proportion, horizontal=True, gap=0.05):
    if False:
        print('Hello World!')
    '\n    Split the given rectangle in n segments whose proportion is specified\n    along the given axis if a gap is inserted, they will be separated by a\n    certain amount of space, retaining the relative proportion between them\n    a gap of 1 correspond to a plot that is half void and the remaining half\n    space is proportionally divided among the pieces.\n    '
    (x, y, w, h) = (float(x), float(y), float(width), float(height))
    if w < 0 or h < 0:
        raise ValueError('dimension of the square less thanzero w={} h=()'.format(w, h))
    proportions = _normalize_split(proportion)
    starting = proportions[:-1]
    amplitude = proportions[1:] - starting
    starting += gap * np.arange(len(proportions) - 1)
    extension = starting[-1] + amplitude[-1] - starting[0]
    starting /= extension
    amplitude /= extension
    starting = (x if horizontal else y) + starting * (w if horizontal else h)
    amplitude = amplitude * (w if horizontal else h)
    results = [(s, y, a, h) if horizontal else (x, s, w, a) for (s, a) in zip(starting, amplitude)]
    return results

def _reduce_dict(count_dict, partial_key):
    if False:
        return 10
    '\n    Make partial sum on a counter dict.\n    Given a match for the beginning of the category, it will sum each value.\n    '
    L = len(partial_key)
    count = sum((v for (k, v) in count_dict.items() if k[:L] == partial_key))
    return count

def _key_splitting(rect_dict, keys, values, key_subset, horizontal, gap):
    if False:
        print('Hello World!')
    '\n    Given a dictionary where each entry  is a rectangle, a list of key and\n    value (count of elements in each category) it split each rect accordingly,\n    as long as the key start with the tuple key_subset.  The other keys are\n    returned without modification.\n    '
    result = {}
    L = len(key_subset)
    for (name, (x, y, w, h)) in rect_dict.items():
        if key_subset == name[:L]:
            divisions = _split_rect(x, y, w, h, values, horizontal, gap)
            for (key, rect) in zip(keys, divisions):
                result[name + (key,)] = rect
        else:
            result[name] = (x, y, w, h)
    return result

def _tuplify(obj):
    if False:
        for i in range(10):
            print('nop')
    'convert an object in a tuple of strings (even if it is not iterable,\n    like a single integer number, but keep the string healthy)\n    '
    if np.iterable(obj) and (not isinstance(obj, str)):
        res = tuple((str(o) for o in obj))
    else:
        res = (str(obj),)
    return res

def _categories_level(keys):
    if False:
        while True:
            i = 10
    'use the Ordered dict to implement a simple ordered set\n    return each level of each category\n    [[key_1_level_1,key_2_level_1],[key_1_level_2,key_2_level_2]]\n    '
    res = []
    for i in zip(*keys):
        tuplefied = _tuplify(i)
        res.append(list(dict([(j, None) for j in tuplefied])))
    return res

def _hierarchical_split(count_dict, horizontal=True, gap=0.05):
    if False:
        return 10
    "\n    Split a square in a hierarchical way given a contingency table.\n\n    Hierarchically split the unit square in alternate directions\n    in proportion to the subdivision contained in the contingency table\n    count_dict.  This is the function that actually perform the tiling\n    for the creation of the mosaic plot.  If the gap array has been specified\n    it will insert a corresponding amount of space (proportional to the\n    unit length), while retaining the proportionality of the tiles.\n\n    Parameters\n    ----------\n    count_dict : dict\n        Dictionary containing the contingency table.\n        Each category should contain a non-negative number\n        with a tuple as index.  It expects that all the combination\n        of keys to be represents; if that is not true, will\n        automatically consider the missing values as 0\n    horizontal : bool\n        The starting direction of the split (by default along\n        the horizontal axis)\n    gap : float or array of floats\n        The list of gaps to be applied on each subdivision.\n        If the length of the given array is less of the number\n        of subcategories (or if it's a single number) it will extend\n        it with exponentially decreasing gaps\n\n    Returns\n    -------\n    base_rect : dict\n        A dictionary containing the result of the split.\n        To each key is associated a 4-tuple of coordinates\n        that are required to create the corresponding rectangle:\n\n            0 - x position of the lower left corner\n            1 - y position of the lower left corner\n            2 - width of the rectangle\n            3 - height of the rectangle\n    "
    base_rect = dict([(tuple(), (0, 0, 1, 1))])
    categories_levels = _categories_level(list(count_dict.keys()))
    L = len(categories_levels)
    if not np.iterable(gap):
        gap = [gap / 1.5 ** idx for idx in range(L)]
    if len(gap) < L:
        last = gap[-1]
        gap = list(*gap) + [last / 1.5 ** idx for idx in range(L)]
    gap = gap[:L]
    count_ordered = dict([(k, count_dict[k]) for k in list(product(*categories_levels))])
    for (cat_idx, cat_enum) in enumerate(categories_levels):
        base_keys = list(product(*categories_levels[:cat_idx]))
        for key in base_keys:
            part_count = [_reduce_dict(count_ordered, key + (partial,)) for partial in cat_enum]
            new_gap = gap[cat_idx]
            base_rect = _key_splitting(base_rect, cat_enum, part_count, key, horizontal, new_gap)
        horizontal = not horizontal
    return base_rect

def _single_hsv_to_rgb(hsv):
    if False:
        print('Hello World!')
    'Transform a color from the hsv space to the rgb.'
    from matplotlib.colors import hsv_to_rgb
    return hsv_to_rgb(array(hsv).reshape(1, 1, 3)).reshape(3)

def _create_default_properties(data):
    if False:
        return 10
    '"Create the default properties of the mosaic given the data\n    first it will varies the color hue (first category) then the color\n    saturation (second category) and then the color value\n    (third category).  If a fourth category is found, it will put\n    decoration on the rectangle.  Does not manage more than four\n    level of categories\n    '
    categories_levels = _categories_level(list(data.keys()))
    Nlevels = len(categories_levels)
    L = len(categories_levels[0])
    hue = np.linspace(0.0, 1.0, L + 2)[:-2]
    L = len(categories_levels[1]) if Nlevels > 1 else 1
    saturation = np.linspace(0.5, 1.0, L + 1)[:-1]
    L = len(categories_levels[2]) if Nlevels > 2 else 1
    value = np.linspace(0.5, 1.0, L + 1)[:-1]
    L = len(categories_levels[3]) if Nlevels > 3 else 1
    hatch = ['', '/', '-', '|', '+'][:L + 1]
    hue = lzip(list(hue), categories_levels[0])
    saturation = lzip(list(saturation), categories_levels[1] if Nlevels > 1 else [''])
    value = lzip(list(value), categories_levels[2] if Nlevels > 2 else [''])
    hatch = lzip(list(hatch), categories_levels[3] if Nlevels > 3 else [''])
    properties = {}
    for (h, s, v, t) in product(hue, saturation, value, hatch):
        (hv, hn) = h
        (sv, sn) = s
        (vv, vn) = v
        (tv, tn) = t
        level = (hn,) + ((sn,) if sn else tuple())
        level = level + ((vn,) if vn else tuple())
        level = level + ((tn,) if tn else tuple())
        hsv = array([hv, sv, vv])
        prop = {'color': _single_hsv_to_rgb(hsv), 'hatch': tv, 'lw': 0}
        properties[level] = prop
    return properties

def _normalize_data(data, index):
    if False:
        for i in range(10):
            print('nop')
    'normalize the data to a dict with tuples of strings as keys\n    right now it works with:\n\n        0 - dictionary (or equivalent mappable)\n        1 - pandas.Series with simple or hierarchical indexes\n        2 - numpy.ndarrays\n        3 - everything that can be converted to a numpy array\n        4 - pandas.DataFrame (via the _normalize_dataframe function)\n    '
    if hasattr(data, 'pivot') and hasattr(data, 'groupby'):
        data = _normalize_dataframe(data, index)
        index = None
    try:
        items = list(data.items())
    except AttributeError:
        data = np.asarray(data)
        temp = {}
        for idx in np.ndindex(data.shape):
            name = tuple((i for i in idx))
            temp[name] = data[idx]
        data = temp
        items = list(data.items())
    data = dict(([_tuplify(k), v] for (k, v) in items))
    categories_levels = _categories_level(list(data.keys()))
    indexes = product(*categories_levels)
    contingency = dict([(k, data.get(k, 0)) for k in indexes])
    data = contingency
    index = lrange(len(categories_levels)) if index is None else index
    contingency = {}
    for (key, value) in data.items():
        new_key = tuple((key[i] for i in index))
        contingency[new_key] = value
    data = contingency
    return data

def _normalize_dataframe(dataframe, index):
    if False:
        return 10
    'Take a pandas DataFrame and count the element present in the\n    given columns, return a hierarchical index on those columns\n    '
    data = dataframe[index].dropna()
    grouped = data.groupby(index, sort=False, observed=False)
    counted = grouped[index].count()
    averaged = counted.mean(axis=1)
    averaged = averaged.fillna(0.0)
    return averaged

def _statistical_coloring(data):
    if False:
        for i in range(10):
            print('nop')
    'evaluate colors from the indipendence properties of the matrix\n    It will encounter problem if one category has all zeros\n    '
    data = _normalize_data(data, None)
    categories_levels = _categories_level(list(data.keys()))
    Nlevels = len(categories_levels)
    total = 1.0 * sum((v for v in data.values()))
    levels_count = []
    for level_idx in range(Nlevels):
        proportion = {}
        for level in categories_levels[level_idx]:
            proportion[level] = 0.0
            for (key, value) in data.items():
                if level == key[level_idx]:
                    proportion[level] += value
            proportion[level] /= total
        levels_count.append(proportion)
    expected = {}
    for (key, value) in data.items():
        base = 1.0
        for (i, k) in enumerate(key):
            base *= levels_count[i][k]
        expected[key] = (base * total, np.sqrt(total * base * (1.0 - base)))
    sigmas = dict(((k, (data[k] - m) / s) for (k, (m, s)) in expected.items()))
    props = {}
    for (key, dev) in sigmas.items():
        red = 0.0 if dev < 0 else dev / (1 + dev)
        blue = 0.0 if dev > 0 else dev / (-1 + dev)
        green = (1.0 - red - blue) / 2.0
        hatch = 'x' if dev > 2 else 'o' if dev < -2 else ''
        props[key] = {'color': [red, green, blue], 'hatch': hatch}
    return props

def _get_position(x, w, h, W):
    if False:
        for i in range(10):
            print('nop')
    if W == 0:
        return x
    return (x + w / 2.0) * w * h / W

def _create_labels(rects, horizontal, ax, rotation):
    if False:
        i = 10
        return i + 15
    'find the position of the label for each value of each category\n\n    right now it supports only up to the four categories\n\n    ax: the axis on which the label should be applied\n    rotation: the rotation list for each side\n    '
    categories = _categories_level(list(rects.keys()))
    if len(categories) > 4:
        msg = 'maximum of 4 level supported for axes labeling... and 4is already a lot of levels, are you sure you need them all?'
        raise ValueError(msg)
    labels = {}
    items = list(rects.items())
    vertical = not horizontal
    ax2 = ax.twinx()
    ax3 = ax.twiny()
    ticks_pos = [ax.set_xticks, ax.set_yticks, ax3.set_xticks, ax2.set_yticks]
    ticks_lab = [ax.set_xticklabels, ax.set_yticklabels, ax3.set_xticklabels, ax2.set_yticklabels]
    if vertical:
        ticks_pos = ticks_pos[1:] + ticks_pos[:1]
        ticks_lab = ticks_lab[1:] + ticks_lab[:1]
    for (pos, lab) in zip(ticks_pos, ticks_lab):
        pos([])
        lab([])
    for (level_idx, level) in enumerate(categories):
        level_ticks = dict()
        for value in level:
            if horizontal:
                if level_idx == 3:
                    index_select = [-1, -1, -1]
                else:
                    index_select = [+0, -1, -1]
            elif level_idx == 3:
                index_select = [+0, -1, +0]
            else:
                index_select = [-1, -1, -1]
            basekey = tuple((categories[i][index_select[i]] for i in range(level_idx)))
            basekey = basekey + (value,)
            subset = dict(((k, v) for (k, v) in items if basekey == k[:level_idx + 1]))
            vals = list(subset.values())
            W = sum((w * h for (x, y, w, h) in vals))
            x_lab = sum((_get_position(x, w, h, W) for (x, y, w, h) in vals))
            y_lab = sum((_get_position(y, h, w, W) for (x, y, w, h) in vals))
            side = (level_idx + vertical) % 4
            level_ticks[value] = y_lab if side % 2 else x_lab
        ticks_pos[level_idx](list(level_ticks.values()))
        ticks_lab[level_idx](list(level_ticks.keys()), rotation=rotation[level_idx])
    return labels

def mosaic(data, index=None, ax=None, horizontal=True, gap=0.005, properties=lambda key: None, labelizer=None, title='', statistic=False, axes_label=True, label_rotation=0.0):
    if False:
        while True:
            i = 10
    "Create a mosaic plot from a contingency table.\n\n    It allows to visualize multivariate categorical data in a rigorous\n    and informative way.\n\n    Parameters\n    ----------\n    data : {dict, Series, ndarray, DataFrame}\n        The contingency table that contains the data.\n        Each category should contain a non-negative number\n        with a tuple as index.  It expects that all the combination\n        of keys to be represents; if that is not true, will\n        automatically consider the missing values as 0.  The order\n        of the keys will be the same as the one of insertion.\n        If a dict of a Series (or any other dict like object)\n        is used, it will take the keys as labels.  If a\n        np.ndarray is provided, it will generate a simple\n        numerical labels.\n    index : list, optional\n        Gives the preferred order for the category ordering. If not specified\n        will default to the given order.  It does not support named indexes\n        for hierarchical Series.  If a DataFrame is provided, it expects\n        a list with the name of the columns.\n    ax : Axes, optional\n        The graph where display the mosaic. If not given, will\n        create a new figure\n    horizontal : bool, optional\n        The starting direction of the split (by default along\n        the horizontal axis)\n    gap : {float, sequence[float]}\n        The list of gaps to be applied on each subdivision.\n        If the length of the given array is less of the number\n        of subcategories (or if it's a single number) it will extend\n        it with exponentially decreasing gaps\n    properties : dict[str, callable], optional\n        A function that for each tile in the mosaic take the key\n        of the tile and returns the dictionary of properties\n        of the generated Rectangle, like color, hatch or similar.\n        A default properties set will be provided fot the keys whose\n        color has not been defined, and will use color variation to help\n        visually separates the various categories. It should return None\n        to indicate that it should use the default property for the tile.\n        A dictionary of the properties for each key can be passed,\n        and it will be internally converted to the correct function\n    labelizer : dict[str, callable], optional\n        A function that generate the text to display at the center of\n        each tile base on the key of that tile\n    title : str, optional\n        The title of the axis\n    statistic : bool, optional\n        If true will use a crude statistical model to give colors to the plot.\n        If the tile has a constraint that is more than 2 standard deviation\n        from the expected value under independence hypothesis, it will\n        go from green to red (for positive deviations, blue otherwise) and\n        will acquire an hatching when crosses the 3 sigma.\n    axes_label : bool, optional\n        Show the name of each value of each category\n        on the axis (default) or hide them.\n    label_rotation : {float, list[float]}\n        The rotation of the axis label (if present). If a list is given\n        each axis can have a different rotation\n\n    Returns\n    -------\n    fig : Figure\n        The figure containing the plot.\n    rects : dict\n        A dictionary that has the same keys of the original\n        dataset, that holds a reference to the coordinates of the\n        tile and the Rectangle that represent it.\n\n    References\n    ----------\n    A Brief History of the Mosaic Display\n        Michael Friendly, York University, Psychology Department\n        Journal of Computational and Graphical Statistics, 2001\n\n    Mosaic Displays for Loglinear Models.\n        Michael Friendly, York University, Psychology Department\n        Proceedings of the Statistical Graphics Section, 1992, 61-68.\n\n    Mosaic displays for multi-way contingency tables.\n        Michael Friendly, York University, Psychology Department\n        Journal of the american statistical association\n        March 1994, Vol. 89, No. 425, Theory and Methods\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> import pandas as pd\n    >>> import matplotlib.pyplot as plt\n    >>> from statsmodels.graphics.mosaicplot import mosaic\n\n    The most simple use case is to take a dictionary and plot the result\n\n    >>> data = {'a': 10, 'b': 15, 'c': 16}\n    >>> mosaic(data, title='basic dictionary')\n    >>> plt.show()\n\n    A more useful example is given by a dictionary with multiple indices.\n    In this case we use a wider gap to a better visual separation of the\n    resulting plot\n\n    >>> data = {('a', 'b'): 1, ('a', 'c'): 2, ('d', 'b'): 3, ('d', 'c'): 4}\n    >>> mosaic(data, gap=0.05, title='complete dictionary')\n    >>> plt.show()\n\n    The same data can be given as a simple or hierarchical indexed Series\n\n    >>> rand = np.random.random\n    >>> from itertools import product\n    >>> tuples = list(product(['bar', 'baz', 'foo', 'qux'], ['one', 'two']))\n    >>> index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])\n    >>> data = pd.Series(rand(8), index=index)\n    >>> mosaic(data, title='hierarchical index series')\n    >>> plt.show()\n\n    The third accepted data structure is the np array, for which a\n    very simple index will be created.\n\n    >>> rand = np.random.random\n    >>> data = 1+rand((2,2))\n    >>> mosaic(data, title='random non-labeled array')\n    >>> plt.show()\n\n    If you need to modify the labeling and the coloring you can give\n    a function tocreate the labels and one with the graphical properties\n    starting from the key tuple\n\n    >>> data = {'a': 10, 'b': 15, 'c': 16}\n    >>> props = lambda key: {'color': 'r' if 'a' in key else 'gray'}\n    >>> labelizer = lambda k: {('a',): 'first', ('b',): 'second',\n    ...                        ('c',): 'third'}[k]\n    >>> mosaic(data, title='colored dictionary', properties=props,\n    ...        labelizer=labelizer)\n    >>> plt.show()\n\n    Using a DataFrame as source, specifying the name of the columns of interest\n\n    >>> gender = ['male', 'male', 'male', 'female', 'female', 'female']\n    >>> pet = ['cat', 'dog', 'dog', 'cat', 'dog', 'cat']\n    >>> data = pd.DataFrame({'gender': gender, 'pet': pet})\n    >>> mosaic(data, ['pet', 'gender'], title='DataFrame as Source')\n    >>> plt.show()\n\n    .. plot :: plots/graphics_mosaicplot_mosaic.py\n    "
    if isinstance(data, DataFrame) and index is None:
        raise ValueError('You must pass an index if data is a DataFrame. See examples.')
    from matplotlib.patches import Rectangle
    (fig, ax) = utils.create_mpl_ax(ax)
    data = _normalize_data(data, index)
    rects = _hierarchical_split(data, horizontal=horizontal, gap=gap)
    if labelizer is None:
        labelizer = lambda k: '\n'.join(k)
    if statistic:
        default_props = _statistical_coloring(data)
    else:
        default_props = _create_default_properties(data)
    if isinstance(properties, dict):
        color_dict = properties
        properties = lambda key: color_dict.get(key, None)
    for (k, v) in rects.items():
        (x, y, w, h) = v
        conf = properties(k)
        props = conf if conf else default_props[k]
        text = labelizer(k)
        Rect = Rectangle((x, y), w, h, label=text, **props)
        ax.add_patch(Rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', size='smaller')
    if axes_label:
        if np.iterable(label_rotation):
            rotation = label_rotation
        else:
            rotation = [label_rotation] * 4
        labels = _create_labels(rects, horizontal, ax, rotation)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
    ax.set_title(title)
    return (fig, rects)