"""
Color schemes for exception handling code in IPython.
"""
import os
from IPython.utils.coloransi import ColorSchemeTable, TermColors, ColorScheme

def exception_colors():
    if False:
        for i in range(10):
            print('nop')
    "Return a color table with fields for exception reporting.\n\n    The table is an instance of ColorSchemeTable with schemes added for\n    'Neutral', 'Linux', 'LightBG' and 'NoColor' and fields for exception handling filled\n    in.\n\n    Examples:\n\n    >>> ec = exception_colors()\n    >>> ec.active_scheme_name\n    ''\n    >>> print(ec.active_colors)\n    None\n\n    Now we activate a color scheme:\n    >>> ec.set_active_scheme('NoColor')\n    >>> ec.active_scheme_name\n    'NoColor'\n    >>> sorted(ec.active_colors.keys())\n    ['Normal', 'caret', 'em', 'excName', 'filename', 'filenameEm', 'line',\n    'lineno', 'linenoEm', 'name', 'nameEm', 'normalEm', 'topline', 'vName',\n    'val', 'valEm']\n    "
    ex_colors = ColorSchemeTable()
    C = TermColors
    ex_colors.add_scheme(ColorScheme('NoColor', topline=C.NoColor, filename=C.NoColor, lineno=C.NoColor, name=C.NoColor, vName=C.NoColor, val=C.NoColor, em=C.NoColor, normalEm=C.NoColor, filenameEm=C.NoColor, linenoEm=C.NoColor, nameEm=C.NoColor, valEm=C.NoColor, excName=C.NoColor, line=C.NoColor, caret=C.NoColor, Normal=C.NoColor))
    ex_colors.add_scheme(ColorScheme('Linux', topline=C.LightRed, filename=C.Green, lineno=C.Green, name=C.Purple, vName=C.Cyan, val=C.Green, em=C.LightCyan, normalEm=C.LightCyan, filenameEm=C.LightGreen, linenoEm=C.LightGreen, nameEm=C.LightPurple, valEm=C.LightBlue, excName=C.LightRed, line=C.Yellow, caret=C.White, Normal=C.Normal))
    ex_colors.add_scheme(ColorScheme('LightBG', topline=C.Red, filename=C.LightGreen, lineno=C.LightGreen, name=C.LightPurple, vName=C.Cyan, val=C.LightGreen, em=C.Cyan, normalEm=C.Cyan, filenameEm=C.Green, linenoEm=C.Green, nameEm=C.Purple, valEm=C.Blue, excName=C.Red, line=C.Red, caret=C.Normal, Normal=C.Normal))
    ex_colors.add_scheme(ColorScheme('Neutral', topline=C.Red, filename=C.LightGreen, lineno=C.LightGreen, name=C.LightPurple, vName=C.Cyan, val=C.LightGreen, em=C.Cyan, normalEm=C.Cyan, filenameEm=C.Green, linenoEm=C.Green, nameEm=C.Purple, valEm=C.Blue, excName=C.Red, line=C.Red, caret=C.Normal, Normal=C.Normal))
    if os.name == 'nt':
        ex_colors.add_scheme(ex_colors['Linux'].copy('Neutral'))
    return ex_colors