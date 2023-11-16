import numpy as np

def rainbow(n):
    if False:
        i = 10
        return i + 15
    '\n    Returns a list of colors sampled at equal intervals over the spectrum.\n\n    Parameters\n    ----------\n    n : int\n        The number of colors to return\n\n    Returns\n    -------\n    R : (n,3) array\n        An of rows of RGB color values\n\n    Notes\n    -----\n    Converts from HSV coordinates (0, 1, 1) to (1, 1, 1) to RGB. Based on\n    the Sage function of the same name.\n    '
    from matplotlib import colors
    R = np.ones((1, n, 3))
    R[0, :, 0] = np.linspace(0, 1, n, endpoint=False)
    return colors.hsv_to_rgb(R).squeeze()