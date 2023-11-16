import os
import re
from subprocess import CalledProcessError
from ..logs import logger
from ..wrappers import run_subprocess

def _get_dpi_from(cmd, pattern, func):
    if False:
        while True:
            i = 10
    'Match pattern against the output of func, passing the results as\n    floats to func.  If anything fails, return None.\n    '
    try:
        (out, _) = run_subprocess([cmd])
    except (OSError, CalledProcessError):
        pass
    else:
        match = re.search(pattern, out)
        if match:
            return func(*map(float, match.groups()))

def _xrandr_calc(x_px, y_px, x_mm, y_mm):
    if False:
        return 10
    if x_mm == 0 or y_mm == 0:
        logger.warning("'xrandr' output has screen dimension of 0mm, " + "can't compute proper DPI")
        return 96.0
    return 25.4 * (x_px / x_mm + y_px / y_mm) / 2

def get_dpi(raise_error=True):
    if False:
        print('Hello World!')
    'Get screen DPI from the OS\n\n    Parameters\n    ----------\n    raise_error : bool\n        If True, raise an error if DPI could not be determined.\n\n    Returns\n    -------\n    dpi : float\n        Dots per inch of the primary screen.\n    '
    if 'DISPLAY' not in os.environ:
        return 96.0
    from_xdpyinfo = _get_dpi_from('xdpyinfo', '(\\d+)x(\\d+) dots per inch', lambda x_dpi, y_dpi: (x_dpi + y_dpi) / 2)
    if from_xdpyinfo is not None:
        return from_xdpyinfo
    from_xrandr = _get_dpi_from('xrandr', '(\\d+)x(\\d+).*?(\\d+)mm x (\\d+)mm', _xrandr_calc)
    if from_xrandr is not None:
        return from_xrandr
    if raise_error:
        raise RuntimeError('could not determine DPI')
    else:
        logger.warning('could not determine DPI')
    return 96