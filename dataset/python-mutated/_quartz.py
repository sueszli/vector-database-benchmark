from ...ext.cocoapy import quartz

def get_dpi(raise_error=True):
    if False:
        i = 10
        return i + 15
    'Get screen DPI from the OS\n\n    Parameters\n    ----------\n    raise_error : bool\n        If True, raise an error if DPI could not be determined.\n\n    Returns\n    -------\n    dpi : float\n        Dots per inch of the primary screen.\n    '
    display = quartz.CGMainDisplayID()
    mm = quartz.CGDisplayScreenSize(display)
    px = quartz.CGDisplayBounds(display).size
    return (px.width / mm.width + px.height / mm.height) * 0.5 * 25.4