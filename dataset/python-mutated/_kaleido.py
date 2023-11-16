import os
import json
from pathlib import Path
import plotly
from plotly.io._utils import validate_coerce_fig_to_dict
try:
    from kaleido.scopes.plotly import PlotlyScope
    scope = PlotlyScope()
    root_dir = os.path.dirname(os.path.abspath(plotly.__file__))
    package_dir = os.path.join(root_dir, 'package_data')
    scope.plotlyjs = os.path.join(package_dir, 'plotly.min.js')
    if scope.mathjax is None:
        scope.mathjax = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js'
except ImportError:
    PlotlyScope = None
    scope = None

def to_image(fig, format=None, width=None, height=None, scale=None, validate=True, engine='auto'):
    if False:
        return 10
    '\n    Convert a figure to a static image bytes string\n\n    Parameters\n    ----------\n    fig:\n        Figure object or dict representing a figure\n\n    format: str or None\n        The desired image format. One of\n          - \'png\'\n          - \'jpg\' or \'jpeg\'\n          - \'webp\'\n          - \'svg\'\n          - \'pdf\'\n          - \'eps\' (Requires the poppler library to be installed and on the PATH)\n\n        If not specified, will default to:\n             - `plotly.io.kaleido.scope.default_format` if engine is "kaleido"\n             - `plotly.io.orca.config.default_format` if engine is "orca"\n\n    width: int or None\n        The width of the exported image in layout pixels. If the `scale`\n        property is 1.0, this will also be the width of the exported image\n        in physical pixels.\n\n        If not specified, will default to:\n             - `plotly.io.kaleido.scope.default_width` if engine is "kaleido"\n             - `plotly.io.orca.config.default_width` if engine is "orca"\n\n    height: int or None\n        The height of the exported image in layout pixels. If the `scale`\n        property is 1.0, this will also be the height of the exported image\n        in physical pixels.\n\n        If not specified, will default to:\n             - `plotly.io.kaleido.scope.default_height` if engine is "kaleido"\n             - `plotly.io.orca.config.default_height` if engine is "orca"\n\n    scale: int or float or None\n        The scale factor to use when exporting the figure. A scale factor\n        larger than 1.0 will increase the image resolution with respect\n        to the figure\'s layout pixel dimensions. Whereas as scale factor of\n        less than 1.0 will decrease the image resolution.\n\n        If not specified, will default to:\n             - `plotly.io.kaleido.scope.default_scale` if engine is "kaleido"\n             - `plotly.io.orca.config.default_scale` if engine is "orca"\n\n\n    validate: bool\n        True if the figure should be validated before being converted to\n        an image, False otherwise.\n\n    engine: str\n        Image export engine to use:\n         - "kaleido": Use Kaleido for image export\n         - "orca": Use Orca for image export\n         - "auto" (default): Use Kaleido if installed, otherwise use orca\n\n    Returns\n    -------\n    bytes\n        The image data\n    '
    if engine == 'auto':
        if scope is not None:
            engine = 'kaleido'
        else:
            from ._orca import validate_executable
            try:
                validate_executable()
                engine = 'orca'
            except:
                engine = 'kaleido'
    if engine == 'orca':
        from ._orca import to_image as to_image_orca
        return to_image_orca(fig, format=format, width=width, height=height, scale=scale, validate=validate)
    elif engine != 'kaleido':
        raise ValueError('Invalid image export engine specified: {engine}'.format(engine=repr(engine)))
    if scope is None:
        raise ValueError('\nImage export using the "kaleido" engine requires the kaleido package,\nwhich can be installed using pip:\n    $ pip install -U kaleido\n')
    fig_dict = validate_coerce_fig_to_dict(fig, validate)
    img_bytes = scope.transform(fig_dict, format=format, width=width, height=height, scale=scale)
    return img_bytes

def write_image(fig, file, format=None, scale=None, width=None, height=None, validate=True, engine='auto'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a figure to a static image and write it to a file or writeable\n    object\n\n    Parameters\n    ----------\n    fig:\n        Figure object or dict representing a figure\n\n    file: str or writeable\n        A string representing a local file path or a writeable object\n        (e.g. a pathlib.Path object or an open file descriptor)\n\n    format: str or None\n        The desired image format. One of\n          - \'png\'\n          - \'jpg\' or \'jpeg\'\n          - \'webp\'\n          - \'svg\'\n          - \'pdf\'\n          - \'eps\' (Requires the poppler library to be installed and on the PATH)\n\n        If not specified and `file` is a string then this will default to the\n        file extension. If not specified and `file` is not a string then this\n        will default to:\n            - `plotly.io.kaleido.scope.default_format` if engine is "kaleido"\n            - `plotly.io.orca.config.default_format` if engine is "orca"\n\n    width: int or None\n        The width of the exported image in layout pixels. If the `scale`\n        property is 1.0, this will also be the width of the exported image\n        in physical pixels.\n\n        If not specified, will default to:\n            - `plotly.io.kaleido.scope.default_width` if engine is "kaleido"\n            - `plotly.io.orca.config.default_width` if engine is "orca"\n\n    height: int or None\n        The height of the exported image in layout pixels. If the `scale`\n        property is 1.0, this will also be the height of the exported image\n        in physical pixels.\n\n        If not specified, will default to:\n            - `plotly.io.kaleido.scope.default_height` if engine is "kaleido"\n            - `plotly.io.orca.config.default_height` if engine is "orca"\n\n    scale: int or float or None\n        The scale factor to use when exporting the figure. A scale factor\n        larger than 1.0 will increase the image resolution with respect\n        to the figure\'s layout pixel dimensions. Whereas as scale factor of\n        less than 1.0 will decrease the image resolution.\n\n        If not specified, will default to:\n            - `plotly.io.kaleido.scope.default_scale` if engine is "kaleido"\n            - `plotly.io.orca.config.default_scale` if engine is "orca"\n\n    validate: bool\n        True if the figure should be validated before being converted to\n        an image, False otherwise.\n\n    engine: str\n        Image export engine to use:\n         - "kaleido": Use Kaleido for image export\n         - "orca": Use Orca for image export\n         - "auto" (default): Use Kaleido if installed, otherwise use orca\n\n    Returns\n    -------\n    None\n    '
    if isinstance(file, str):
        path = Path(file)
    elif isinstance(file, Path):
        path = file
    else:
        path = None
    if path is not None and format is None:
        ext = path.suffix
        if ext:
            format = ext.lstrip('.')
        else:
            raise ValueError("\nCannot infer image type from output path '{file}'.\nPlease add a file extension or specify the type using the format parameter.\nFor example:\n\n    >>> import plotly.io as pio\n    >>> pio.write_image(fig, file_path, format='png')\n".format(file=file))
    img_data = to_image(fig, format=format, scale=scale, width=width, height=height, validate=validate, engine=engine)
    if path is None:
        try:
            file.write(img_data)
            return
        except AttributeError:
            pass
        raise ValueError("\nThe 'file' argument '{file}' is not a string, pathlib.Path object, or file descriptor.\n".format(file=file))
    else:
        path.write_bytes(img_data)

def full_figure_for_development(fig, warn=True, as_dict=False):
    if False:
        return 10
    '\n    Compute default values for all attributes not specified in the input figure and\n    returns the output as a "full" figure. This function calls Plotly.js via Kaleido\n    to populate unspecified attributes. This function is intended for interactive use\n    during development to learn more about how Plotly.js computes default values and is\n    not generally necessary or recommended for production use.\n\n    Parameters\n    ----------\n    fig:\n        Figure object or dict representing a figure\n\n    warn: bool\n        If False, suppress warnings about not using this in production.\n\n    as_dict: bool\n        If True, output is a dict with some keys that go.Figure can\'t parse.\n        If False, output is a go.Figure with unparseable keys skipped.\n\n    Returns\n    -------\n    plotly.graph_objects.Figure or dict\n        The full figure\n    '
    if scope is None:
        raise ValueError('\nFull figure generation requires the kaleido package,\nwhich can be installed using pip:\n    $ pip install -U kaleido\n')
    if warn:
        import warnings
        warnings.warn('full_figure_for_development is not recommended or necessary for production use in most circumstances. \nTo suppress this warning, set warn=False')
    fig = json.loads(scope.transform(fig, format='json').decode('utf-8'))
    if as_dict:
        return fig
    else:
        import plotly.graph_objects as go
        return go.Figure(fig, skip_invalid=True)
__all__ = ['to_image', 'write_image', 'scope', 'full_figure_for_development']