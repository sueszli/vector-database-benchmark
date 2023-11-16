"""Tools for setting up printing in interactive sessions. """
from sympy.external.importtools import version_tuple
from io import BytesIO
from sympy.printing.latex import latex as default_latex
from sympy.printing.preview import preview
from sympy.utilities.misc import debug
from sympy.printing.defaults import Printable

def _init_python_printing(stringify_func, **settings):
    if False:
        for i in range(10):
            print('nop')
    'Setup printing in Python interactive session. '
    import sys
    import builtins

    def _displayhook(arg):
        if False:
            while True:
                i = 10
        "Python's pretty-printer display hook.\n\n           This function was adapted from:\n\n            https://www.python.org/dev/peps/pep-0217/\n\n        "
        if arg is not None:
            builtins._ = None
            print(stringify_func(arg, **settings))
            builtins._ = arg
    sys.displayhook = _displayhook

def _init_ipython_printing(ip, stringify_func, use_latex, euler, forecolor, backcolor, fontsize, latex_mode, print_builtin, latex_printer, scale, **settings):
    if False:
        return 10
    'Setup printing in IPython interactive session. '
    try:
        from IPython.lib.latextools import latex_to_png
    except ImportError:
        pass
    if forecolor is None:
        color = ip.colors.lower()
        if color == 'lightbg':
            forecolor = 'Black'
        elif color == 'linux':
            forecolor = 'White'
        else:
            forecolor = 'Gray'
        debug('init_printing: Automatic foreground color:', forecolor)
    if use_latex == 'svg':
        extra_preamble = '\n\\special{color %s}' % forecolor
    else:
        extra_preamble = ''
    imagesize = 'tight'
    offset = '0cm,0cm'
    resolution = round(150 * scale)
    dvi = '-T %s -D %d -bg %s -fg %s -O %s' % (imagesize, resolution, backcolor, forecolor, offset)
    dvioptions = dvi.split()
    svg_scale = 150 / 72 * scale
    dvioptions_svg = ['--no-fonts', '--scale={}'.format(svg_scale)]
    debug('init_printing: DVIOPTIONS:', dvioptions)
    debug('init_printing: DVIOPTIONS_SVG:', dvioptions_svg)
    latex = latex_printer or default_latex

    def _print_plain(arg, p, cycle):
        if False:
            while True:
                i = 10
        'caller for pretty, for use in IPython 0.11'
        if _can_print(arg):
            p.text(stringify_func(arg))
        else:
            p.text(IPython.lib.pretty.pretty(arg))

    def _preview_wrapper(o):
        if False:
            while True:
                i = 10
        exprbuffer = BytesIO()
        try:
            preview(o, output='png', viewer='BytesIO', euler=euler, outputbuffer=exprbuffer, extra_preamble=extra_preamble, dvioptions=dvioptions, fontsize=fontsize)
        except Exception as e:
            debug('png printing:', '_preview_wrapper exception raised:', repr(e))
            raise
        return exprbuffer.getvalue()

    def _svg_wrapper(o):
        if False:
            i = 10
            return i + 15
        exprbuffer = BytesIO()
        try:
            preview(o, output='svg', viewer='BytesIO', euler=euler, outputbuffer=exprbuffer, extra_preamble=extra_preamble, dvioptions=dvioptions_svg, fontsize=fontsize)
        except Exception as e:
            debug('svg printing:', '_preview_wrapper exception raised:', repr(e))
            raise
        return exprbuffer.getvalue().decode('utf-8')

    def _matplotlib_wrapper(o):
        if False:
            print('Hello World!')
        try:
            try:
                return latex_to_png(o, color=forecolor, scale=scale)
            except TypeError:
                return latex_to_png(o)
        except ValueError as e:
            debug('matplotlib exception caught:', repr(e))
            return None
    printing_hooks = ('_latex', '_sympystr', '_pretty', '_sympyrepr')

    def _can_print(o):
        if False:
            while True:
                i = 10
        'Return True if type o can be printed with one of the SymPy printers.\n\n        If o is a container type, this is True if and only if every element of\n        o can be printed in this way.\n        '
        try:
            builtin_types = (list, tuple, set, frozenset)
            if isinstance(o, builtin_types):
                if type(o).__str__ not in (i.__str__ for i in builtin_types) or type(o).__repr__ not in (i.__repr__ for i in builtin_types):
                    return False
                return all((_can_print(i) for i in o))
            elif isinstance(o, dict):
                return all((_can_print(i) and _can_print(o[i]) for i in o))
            elif isinstance(o, bool):
                return False
            elif isinstance(o, Printable):
                return True
            elif any((hasattr(o, hook) for hook in printing_hooks)):
                return True
            elif isinstance(o, (float, int)) and print_builtin:
                return True
            return False
        except RuntimeError:
            return False

    def _print_latex_png(o):
        if False:
            while True:
                i = 10
        '\n        A function that returns a png rendered by an external latex\n        distribution, falling back to matplotlib rendering\n        '
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                s = '$\\displaystyle %s$' % s
            try:
                return _preview_wrapper(s)
            except RuntimeError as e:
                debug('preview failed with:', repr(e), ' Falling back to matplotlib backend')
                if latex_mode != 'inline':
                    s = latex(o, mode='inline', **settings)
                return _matplotlib_wrapper(s)

    def _print_latex_svg(o):
        if False:
            while True:
                i = 10
        '\n        A function that returns a svg rendered by an external latex\n        distribution, no fallback available.\n        '
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                s = '$\\displaystyle %s$' % s
            try:
                return _svg_wrapper(s)
            except RuntimeError as e:
                debug('preview failed with:', repr(e), ' No fallback available.')

    def _print_latex_matplotlib(o):
        if False:
            i = 10
            return i + 15
        '\n        A function that returns a png rendered by mathtext\n        '
        if _can_print(o):
            s = latex(o, mode='inline', **settings)
            return _matplotlib_wrapper(s)

    def _print_latex_text(o):
        if False:
            while True:
                i = 10
        '\n        A function to generate the latex representation of SymPy expressions.\n        '
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                return '$\\displaystyle %s$' % s
            return s

    def _result_display(self, arg):
        if False:
            for i in range(10):
                print('nop')
        "IPython's pretty-printer display hook, for use in IPython 0.10\n\n           This function was adapted from:\n\n            ipython/IPython/hooks.py:155\n\n        "
        if self.rc.pprint:
            out = stringify_func(arg)
            if '\n' in out:
                print()
            print(out)
        else:
            print(repr(arg))
    import IPython
    if version_tuple(IPython.__version__) >= version_tuple('0.11'):
        printable_types = [float, tuple, list, set, frozenset, dict, int]
        plaintext_formatter = ip.display_formatter.formatters['text/plain']
        for cls in printable_types + [Printable]:
            plaintext_formatter.for_type(cls, _print_plain)
        svg_formatter = ip.display_formatter.formatters['image/svg+xml']
        if use_latex in ('svg',):
            debug('init_printing: using svg formatter')
            for cls in printable_types:
                svg_formatter.for_type(cls, _print_latex_svg)
            Printable._repr_svg_ = _print_latex_svg
        else:
            debug('init_printing: not using any svg formatter')
            for cls in printable_types:
                if cls in svg_formatter.type_printers:
                    svg_formatter.type_printers.pop(cls)
            Printable._repr_svg_ = Printable._repr_disabled
        png_formatter = ip.display_formatter.formatters['image/png']
        if use_latex in (True, 'png'):
            debug('init_printing: using png formatter')
            for cls in printable_types:
                png_formatter.for_type(cls, _print_latex_png)
            Printable._repr_png_ = _print_latex_png
        elif use_latex == 'matplotlib':
            debug('init_printing: using matplotlib formatter')
            for cls in printable_types:
                png_formatter.for_type(cls, _print_latex_matplotlib)
            Printable._repr_png_ = _print_latex_matplotlib
        else:
            debug('init_printing: not using any png formatter')
            for cls in printable_types:
                if cls in png_formatter.type_printers:
                    png_formatter.type_printers.pop(cls)
            Printable._repr_png_ = Printable._repr_disabled
        latex_formatter = ip.display_formatter.formatters['text/latex']
        if use_latex in (True, 'mathjax'):
            debug('init_printing: using mathjax formatter')
            for cls in printable_types:
                latex_formatter.for_type(cls, _print_latex_text)
            Printable._repr_latex_ = _print_latex_text
        else:
            debug('init_printing: not using text/latex formatter')
            for cls in printable_types:
                if cls in latex_formatter.type_printers:
                    latex_formatter.type_printers.pop(cls)
            Printable._repr_latex_ = Printable._repr_disabled
    else:
        ip.set_hook('result_display', _result_display)

def _is_ipython(shell):
    if False:
        for i in range(10):
            print('nop')
    'Is a shell instance an IPython shell?'
    from sys import modules
    if 'IPython' not in modules:
        return False
    try:
        from IPython.core.interactiveshell import InteractiveShell
    except ImportError:
        try:
            from IPython.iplib import InteractiveShell
        except ImportError:
            return False
    return isinstance(shell, InteractiveShell)
NO_GLOBAL = False

def init_printing(pretty_print=True, order=None, use_unicode=None, use_latex=None, wrap_line=None, num_columns=None, no_global=False, ip=None, euler=False, forecolor=None, backcolor='Transparent', fontsize='10pt', latex_mode='plain', print_builtin=True, str_printer=None, pretty_printer=None, latex_printer=None, scale=1.0, **settings):
    if False:
        i = 10
        return i + 15
    "\n    Initializes pretty-printer depending on the environment.\n\n    Parameters\n    ==========\n\n    pretty_print : bool, default=True\n        If ``True``, use :func:`~.pretty_print` to stringify or the provided pretty\n        printer; if ``False``, use :func:`~.sstrrepr` to stringify or the provided string\n        printer.\n    order : string or None, default='lex'\n        There are a few different settings for this parameter:\n        ``'lex'`` (default), which is lexographic order;\n        ``'grlex'``, which is graded lexographic order;\n        ``'grevlex'``, which is reversed graded lexographic order;\n        ``'old'``, which is used for compatibility reasons and for long expressions;\n        ``None``, which sets it to lex.\n    use_unicode : bool or None, default=None\n        If ``True``, use unicode characters;\n        if ``False``, do not use unicode characters;\n        if ``None``, make a guess based on the environment.\n    use_latex : string, bool, or None, default=None\n        If ``True``, use default LaTeX rendering in GUI interfaces (png and\n        mathjax);\n        if ``False``, do not use LaTeX rendering;\n        if ``None``, make a guess based on the environment;\n        if ``'png'``, enable LaTeX rendering with an external LaTeX compiler,\n        falling back to matplotlib if external compilation fails;\n        if ``'matplotlib'``, enable LaTeX rendering with matplotlib;\n        if ``'mathjax'``, enable LaTeX text generation, for example MathJax\n        rendering in IPython notebook or text rendering in LaTeX documents;\n        if ``'svg'``, enable LaTeX rendering with an external latex compiler,\n        no fallback\n    wrap_line : bool\n        If True, lines will wrap at the end; if False, they will not wrap\n        but continue as one line. This is only relevant if ``pretty_print`` is\n        True.\n    num_columns : int or None, default=None\n        If ``int``, number of columns before wrapping is set to num_columns; if\n        ``None``, number of columns before wrapping is set to terminal width.\n        This is only relevant if ``pretty_print`` is ``True``.\n    no_global : bool, default=False\n        If ``True``, the settings become system wide;\n        if ``False``, use just for this console/session.\n    ip : An interactive console\n        This can either be an instance of IPython,\n        or a class that derives from code.InteractiveConsole.\n    euler : bool, optional, default=False\n        Loads the euler package in the LaTeX preamble for handwritten style\n        fonts (https://www.ctan.org/pkg/euler).\n    forecolor : string or None, optional, default=None\n        DVI setting for foreground color. ``None`` means that either ``'Black'``,\n        ``'White'``, or ``'Gray'`` will be selected based on a guess of the IPython\n        terminal color setting. See notes.\n    backcolor : string, optional, default='Transparent'\n        DVI setting for background color. See notes.\n    fontsize : string or int, optional, default='10pt'\n        A font size to pass to the LaTeX documentclass function in the\n        preamble. Note that the options are limited by the documentclass.\n        Consider using scale instead.\n    latex_mode : string, optional, default='plain'\n        The mode used in the LaTeX printer. Can be one of:\n        ``{'inline'|'plain'|'equation'|'equation*'}``.\n    print_builtin : boolean, optional, default=True\n        If ``True`` then floats and integers will be printed. If ``False`` the\n        printer will only print SymPy types.\n    str_printer : function, optional, default=None\n        A custom string printer function. This should mimic\n        :func:`~.sstrrepr()`.\n    pretty_printer : function, optional, default=None\n        A custom pretty printer. This should mimic :func:`~.pretty()`.\n    latex_printer : function, optional, default=None\n        A custom LaTeX printer. This should mimic :func:`~.latex()`.\n    scale : float, optional, default=1.0\n        Scale the LaTeX output when using the ``'png'`` or ``'svg'`` backends.\n        Useful for high dpi screens.\n    settings :\n        Any additional settings for the ``latex`` and ``pretty`` commands can\n        be used to fine-tune the output.\n\n    Examples\n    ========\n\n    >>> from sympy.interactive import init_printing\n    >>> from sympy import Symbol, sqrt\n    >>> from sympy.abc import x, y\n    >>> sqrt(5)\n    sqrt(5)\n    >>> init_printing(pretty_print=True) # doctest: +SKIP\n    >>> sqrt(5) # doctest: +SKIP\n      ___\n    \\/ 5\n    >>> theta = Symbol('theta') # doctest: +SKIP\n    >>> init_printing(use_unicode=True) # doctest: +SKIP\n    >>> theta # doctest: +SKIP\n    \\u03b8\n    >>> init_printing(use_unicode=False) # doctest: +SKIP\n    >>> theta # doctest: +SKIP\n    theta\n    >>> init_printing(order='lex') # doctest: +SKIP\n    >>> str(y + x + y**2 + x**2) # doctest: +SKIP\n    x**2 + x + y**2 + y\n    >>> init_printing(order='grlex') # doctest: +SKIP\n    >>> str(y + x + y**2 + x**2) # doctest: +SKIP\n    x**2 + x + y**2 + y\n    >>> init_printing(order='grevlex') # doctest: +SKIP\n    >>> str(y * x**2 + x * y**2) # doctest: +SKIP\n    x**2*y + x*y**2\n    >>> init_printing(order='old') # doctest: +SKIP\n    >>> str(x**2 + y**2 + x + y) # doctest: +SKIP\n    x**2 + x + y**2 + y\n    >>> init_printing(num_columns=10) # doctest: +SKIP\n    >>> x**2 + x + y**2 + y # doctest: +SKIP\n    x + y +\n    x**2 + y**2\n\n    Notes\n    =====\n\n    The foreground and background colors can be selected when using ``'png'`` or\n    ``'svg'`` LaTeX rendering. Note that before the ``init_printing`` command is\n    executed, the LaTeX rendering is handled by the IPython console and not SymPy.\n\n    The colors can be selected among the 68 standard colors known to ``dvips``,\n    for a list see [1]_. In addition, the background color can be\n    set to  ``'Transparent'`` (which is the default value).\n\n    When using the ``'Auto'`` foreground color, the guess is based on the\n    ``colors`` variable in the IPython console, see [2]_. Hence, if\n    that variable is set correctly in your IPython console, there is a high\n    chance that the output will be readable, although manual settings may be\n    needed.\n\n\n    References\n    ==========\n\n    .. [1] https://en.wikibooks.org/wiki/LaTeX/Colors#The_68_standard_colors_known_to_dvips\n\n    .. [2] https://ipython.readthedocs.io/en/stable/config/details.html#terminal-colors\n\n    See Also\n    ========\n\n    sympy.printing.latex\n    sympy.printing.pretty\n\n    "
    import sys
    from sympy.printing.printer import Printer
    if pretty_print:
        if pretty_printer is not None:
            stringify_func = pretty_printer
        else:
            from sympy.printing import pretty as stringify_func
    elif str_printer is not None:
        stringify_func = str_printer
    else:
        from sympy.printing import sstrrepr as stringify_func
    in_ipython = False
    if ip is None:
        try:
            ip = get_ipython()
        except NameError:
            pass
        else:
            in_ipython = ip is not None
    if ip and (not in_ipython):
        in_ipython = _is_ipython(ip)
    if in_ipython and pretty_print:
        try:
            import IPython
            if version_tuple(IPython.__version__) >= version_tuple('1.0'):
                from IPython.terminal.interactiveshell import TerminalInteractiveShell
            else:
                from IPython.frontend.terminal.interactiveshell import TerminalInteractiveShell
            from code import InteractiveConsole
        except ImportError:
            pass
        else:
            if not isinstance(ip, (InteractiveConsole, TerminalInteractiveShell)) and 'ipython-console' not in ''.join(sys.argv):
                if use_unicode is None:
                    debug('init_printing: Setting use_unicode to True')
                    use_unicode = True
                if use_latex is None:
                    debug('init_printing: Setting use_latex to True')
                    use_latex = True
    if not NO_GLOBAL and (not no_global):
        Printer.set_global_settings(order=order, use_unicode=use_unicode, wrap_line=wrap_line, num_columns=num_columns)
    else:
        _stringify_func = stringify_func
        if pretty_print:
            stringify_func = lambda expr, **settings: _stringify_func(expr, order=order, use_unicode=use_unicode, wrap_line=wrap_line, num_columns=num_columns, **settings)
        else:
            stringify_func = lambda expr, **settings: _stringify_func(expr, order=order, **settings)
    if in_ipython:
        mode_in_settings = settings.pop('mode', None)
        if mode_in_settings:
            debug('init_printing: Mode is not able to be set due to internalsof IPython printing')
        _init_ipython_printing(ip, stringify_func, use_latex, euler, forecolor, backcolor, fontsize, latex_mode, print_builtin, latex_printer, scale, **settings)
    else:
        _init_python_printing(stringify_func, **settings)