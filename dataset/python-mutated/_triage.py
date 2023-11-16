import sys
from ._vispy_fonts import _vispy_fonts
if sys.platform.startswith('linux'):
    from ._freetype import _load_glyph
    from ...ext.fontconfig import _list_fonts
elif sys.platform == 'darwin':
    from ._quartz import _load_glyph, _list_fonts
elif sys.platform.startswith('win'):
    from ._freetype import _load_glyph
    from ._win32 import _list_fonts
else:
    raise NotImplementedError('unknown system %s' % sys.platform)
_fonts = {}

def list_fonts():
    if False:
        for i in range(10):
            print('nop')
    'List system fonts\n\n    Returns\n    -------\n    fonts : list of str\n        List of system fonts.\n    '
    vals = _list_fonts()
    for font in _vispy_fonts:
        vals += [font] if font not in vals else []
    vals = sorted(vals, key=lambda s: s.lower())
    return vals