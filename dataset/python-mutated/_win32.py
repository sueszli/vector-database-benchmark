from ...ext.gdi32plus import gdi32, user32, HORZSIZE, VERTSIZE, HORZRES, VERTRES

def get_dpi(raise_error=True):
    if False:
        while True:
            i = 10
    'Get screen DPI from the OS\n\n    Parameters\n    ----------\n    raise_error : bool\n        If True, raise an error if DPI could not be determined.\n\n    Returns\n    -------\n    dpi : float\n        Dots per inch of the primary screen.\n    '
    try:
        user32.SetProcessDPIAware()
    except AttributeError:
        pass
    dc = user32.GetDC(0)
    h_size = gdi32.GetDeviceCaps(dc, HORZSIZE)
    v_size = gdi32.GetDeviceCaps(dc, VERTSIZE)
    h_res = gdi32.GetDeviceCaps(dc, HORZRES)
    v_res = gdi32.GetDeviceCaps(dc, VERTRES)
    user32.ReleaseDC(None, dc)
    return (h_res / float(h_size) + v_res / float(v_size)) * 0.5 * 25.4