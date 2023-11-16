"""
Color themes for the interactive console.
"""
import html
import sys
from typing import Any, List, Optional, Tuple, cast
from scapy.compat import Protocol

class ColorTable:
    colors = {'black': ('\x1b[30m', '#ansiblack'), 'red': ('\x1b[31m', '#ansired'), 'green': ('\x1b[32m', '#ansigreen'), 'yellow': ('\x1b[33m', '#ansiyellow'), 'blue': ('\x1b[34m', '#ansiblue'), 'purple': ('\x1b[35m', '#ansipurple'), 'cyan': ('\x1b[36m', '#ansicyan'), 'grey': ('\x1b[37m', '#ansiwhite'), 'reset': ('\x1b[39m', 'noinherit'), 'bg_black': ('\x1b[40m', 'bg:#ansiblack'), 'bg_red': ('\x1b[41m', 'bg:#ansired'), 'bg_green': ('\x1b[42m', 'bg:#ansigreen'), 'bg_yellow': ('\x1b[43m', 'bg:#ansiyellow'), 'bg_blue': ('\x1b[44m', 'bg:#ansiblue'), 'bg_purple': ('\x1b[45m', 'bg:#ansipurple'), 'bg_cyan': ('\x1b[46m', 'bg:#ansicyan'), 'bg_grey': ('\x1b[47m', 'bg:#ansiwhite'), 'bg_reset': ('\x1b[49m', 'noinherit'), 'normal': ('\x1b[0m', 'noinherit'), 'bold': ('\x1b[1m', 'bold'), 'uline': ('\x1b[4m', 'underline'), 'blink': ('\x1b[5m', ''), 'invert': ('\x1b[7m', '')}
    inv_map = {v[0]: v[1] for (k, v) in colors.items()}

    def __repr__(self):
        if False:
            return 10
        return '<ColorTable>'

    def __getattr__(self, attr):
        if False:
            print('Hello World!')
        return self.colors.get(attr, [''])[0]

    def ansi_to_pygments(self, x):
        if False:
            print('Hello World!')
        '\n        Transform ansi encoded text to Pygments text\n        '
        for (k, v) in self.inv_map.items():
            x = x.replace(k, ' ' + v)
        return x.strip()
Color = ColorTable()

class _ColorFormatterType(Protocol):

    def __call__(self, val: Any, fmt: Optional[str]=None, fmt2: str='', before: str='', after: str='') -> str:
        if False:
            print('Hello World!')
        pass

def create_styler(fmt=None, before='', after='', fmt2='%s'):
    if False:
        for i in range(10):
            print('nop')

    def do_style(val: Any, fmt: Optional[str]=fmt, fmt2: str=fmt2, before: str=before, after: str=after) -> str:
        if False:
            return 10
        if fmt is None:
            sval = str(val)
        else:
            sval = fmt % val
        return fmt2 % (before + sval + after)
    return do_style

class ColorTheme:
    style_normal = ''
    style_prompt = ''
    style_punct = ''
    style_id = ''
    style_not_printable = ''
    style_layer_name = ''
    style_field_name = ''
    style_field_value = ''
    style_emph_field_name = ''
    style_emph_field_value = ''
    style_packetlist_name = ''
    style_packetlist_proto = ''
    style_packetlist_value = ''
    style_fail = ''
    style_success = ''
    style_odd = ''
    style_even = ''
    style_opening = ''
    style_active = ''
    style_closed = ''
    style_left = ''
    style_right = ''
    style_logo = ''

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<%s>' % self.__class__.__name__

    def __reduce__(self):
        if False:
            print('Hello World!')
        return (self.__class__, (), ())

    def __getattr__(self, attr):
        if False:
            for i in range(10):
                print('nop')
        if attr in ['__getstate__', '__setstate__', '__getinitargs__', '__reduce_ex__']:
            raise AttributeError()
        return create_styler()

    def format(self, string, fmt):
        if False:
            print('Hello World!')
        for style in fmt.split('+'):
            string = getattr(self, style)(string)
        return string

class NoTheme(ColorTheme):
    pass

class AnsiColorTheme(ColorTheme):

    def __getattr__(self, attr):
        if False:
            print('Hello World!')
        if attr.startswith('__'):
            raise AttributeError(attr)
        s = 'style_%s' % attr
        if s in self.__class__.__dict__:
            before = getattr(self, s)
            after = self.style_normal
        elif not isinstance(self, BlackAndWhite) and attr in Color.colors:
            before = Color.colors[attr][0]
            after = Color.colors['normal'][0]
        else:
            before = after = ''
        return create_styler(before=before, after=after)

class BlackAndWhite(AnsiColorTheme, NoTheme):
    pass

class DefaultTheme(AnsiColorTheme):
    style_normal = Color.normal
    style_prompt = Color.blue + Color.bold
    style_punct = Color.normal
    style_id = Color.blue + Color.bold
    style_not_printable = Color.grey
    style_layer_name = Color.red + Color.bold
    style_field_name = Color.blue
    style_field_value = Color.purple
    style_emph_field_name = Color.blue + Color.uline + Color.bold
    style_emph_field_value = Color.purple + Color.uline + Color.bold
    style_packetlist_name = Color.red + Color.bold
    style_packetlist_proto = Color.blue
    style_packetlist_value = Color.purple
    style_fail = Color.red + Color.bold
    style_success = Color.blue + Color.bold
    style_even = Color.black + Color.bold
    style_odd = Color.black
    style_opening = Color.yellow
    style_active = Color.black
    style_closed = Color.grey
    style_left = Color.blue + Color.invert
    style_right = Color.red + Color.invert
    style_logo = Color.green + Color.bold

class BrightTheme(AnsiColorTheme):
    style_normal = Color.normal
    style_punct = Color.normal
    style_id = Color.yellow + Color.bold
    style_layer_name = Color.red + Color.bold
    style_field_name = Color.yellow + Color.bold
    style_field_value = Color.purple + Color.bold
    style_emph_field_name = Color.yellow + Color.bold
    style_emph_field_value = Color.green + Color.bold
    style_packetlist_name = Color.red + Color.bold
    style_packetlist_proto = Color.yellow + Color.bold
    style_packetlist_value = Color.purple + Color.bold
    style_fail = Color.red + Color.bold
    style_success = Color.blue + Color.bold
    style_even = Color.black + Color.bold
    style_odd = Color.black
    style_left = Color.cyan + Color.invert
    style_right = Color.purple + Color.invert
    style_logo = Color.green + Color.bold

class RastaTheme(AnsiColorTheme):
    style_normal = Color.normal + Color.green + Color.bold
    style_prompt = Color.yellow + Color.bold
    style_punct = Color.red
    style_id = Color.green + Color.bold
    style_not_printable = Color.green
    style_layer_name = Color.red + Color.bold
    style_field_name = Color.yellow + Color.bold
    style_field_value = Color.green + Color.bold
    style_emph_field_name = Color.green
    style_emph_field_value = Color.green
    style_packetlist_name = Color.red + Color.bold
    style_packetlist_proto = Color.yellow + Color.bold
    style_packetlist_value = Color.green + Color.bold
    style_fail = Color.red
    style_success = Color.red + Color.bold
    style_even = Color.yellow
    style_odd = Color.green
    style_left = Color.yellow + Color.invert
    style_right = Color.red + Color.invert
    style_logo = Color.green + Color.bold

class ColorOnBlackTheme(AnsiColorTheme):
    """Color theme for black backgrounds"""
    style_normal = Color.normal
    style_prompt = Color.green + Color.bold
    style_punct = Color.normal
    style_id = Color.green
    style_not_printable = Color.black + Color.bold
    style_layer_name = Color.yellow + Color.bold
    style_field_name = Color.cyan
    style_field_value = Color.purple + Color.bold
    style_emph_field_name = Color.cyan + Color.bold
    style_emph_field_value = Color.red + Color.bold
    style_packetlist_name = Color.black + Color.bold
    style_packetlist_proto = Color.yellow + Color.bold
    style_packetlist_value = Color.purple + Color.bold
    style_fail = Color.red + Color.bold
    style_success = Color.green
    style_even = Color.black + Color.bold
    style_odd = Color.grey
    style_opening = Color.yellow
    style_active = Color.grey + Color.bold
    style_closed = Color.black + Color.bold
    style_left = Color.cyan + Color.bold
    style_right = Color.red + Color.bold
    style_logo = Color.green + Color.bold

class FormatTheme(ColorTheme):

    def __getattr__(self, attr: str) -> _ColorFormatterType:
        if False:
            for i in range(10):
                print('nop')
        if attr.startswith('__'):
            raise AttributeError(attr)
        colfmt = self.__class__.__dict__.get('style_%s' % attr, '%s')
        return create_styler(fmt2=colfmt)

class LatexTheme(FormatTheme):
    """
    You can prepend the output from this theme with
    \\tt\\obeyspaces\\obeylines\\tiny\\noindent
    """
    style_prompt = '\\textcolor{blue}{%s}'
    style_not_printable = '\\textcolor{gray}{%s}'
    style_layer_name = '\\textcolor{red}{\\bf %s}'
    style_field_name = '\\textcolor{blue}{%s}'
    style_field_value = '\\textcolor{purple}{%s}'
    style_emph_field_name = '\\textcolor{blue}{\\underline{%s}}'
    style_emph_field_value = '\\textcolor{purple}{\\underline{%s}}'
    style_packetlist_name = '\\textcolor{red}{\\bf %s}'
    style_packetlist_proto = '\\textcolor{blue}{%s}'
    style_packetlist_value = '\\textcolor{purple}{%s}'
    style_fail = '\\textcolor{red}{\\bf %s}'
    style_success = '\\textcolor{blue}{\\bf %s}'
    style_left = '\\textcolor{blue}{%s}'
    style_right = '\\textcolor{red}{%s}'
    style_logo = '\\textcolor{green}{\\bf %s}'

    def __getattr__(self, attr: str) -> _ColorFormatterType:
        if False:
            print('Hello World!')
        from scapy.utils import tex_escape
        styler = super(LatexTheme, self).__getattr__(attr)
        return cast(_ColorFormatterType, lambda x, *args, **kwargs: styler(tex_escape(x), *args, **kwargs))

class LatexTheme2(FormatTheme):
    style_prompt = '@`@textcolor@[@blue@]@@[@%s@]@'
    style_not_printable = '@`@textcolor@[@gray@]@@[@%s@]@'
    style_layer_name = '@`@textcolor@[@red@]@@[@@`@bfseries@[@@]@%s@]@'
    style_field_name = '@`@textcolor@[@blue@]@@[@%s@]@'
    style_field_value = '@`@textcolor@[@purple@]@@[@%s@]@'
    style_emph_field_name = '@`@textcolor@[@blue@]@@[@@`@underline@[@%s@]@@]@'
    style_emph_field_value = '@`@textcolor@[@purple@]@@[@@`@underline@[@%s@]@@]@'
    style_packetlist_name = '@`@textcolor@[@red@]@@[@@`@bfseries@[@@]@%s@]@'
    style_packetlist_proto = '@`@textcolor@[@blue@]@@[@%s@]@'
    style_packetlist_value = '@`@textcolor@[@purple@]@@[@%s@]@'
    style_fail = '@`@textcolor@[@red@]@@[@@`@bfseries@[@@]@%s@]@'
    style_success = '@`@textcolor@[@blue@]@@[@@`@bfseries@[@@]@%s@]@'
    style_even = '@`@textcolor@[@gray@]@@[@@`@bfseries@[@@]@%s@]@'
    style_left = '@`@textcolor@[@blue@]@@[@%s@]@'
    style_right = '@`@textcolor@[@red@]@@[@%s@]@'
    style_logo = '@`@textcolor@[@green@]@@[@@`@bfseries@[@@]@%s@]@'

class HTMLTheme(FormatTheme):
    style_prompt = '<span class=prompt>%s</span>'
    style_not_printable = '<span class=not_printable>%s</span>'
    style_layer_name = '<span class=layer_name>%s</span>'
    style_field_name = '<span class=field_name>%s</span>'
    style_field_value = '<span class=field_value>%s</span>'
    style_emph_field_name = '<span class=emph_field_name>%s</span>'
    style_emph_field_value = '<span class=emph_field_value>%s</span>'
    style_packetlist_name = '<span class=packetlist_name>%s</span>'
    style_packetlist_proto = '<span class=packetlist_proto>%s</span>'
    style_packetlist_value = '<span class=packetlist_value>%s</span>'
    style_fail = '<span class=fail>%s</span>'
    style_success = '<span class=success>%s</span>'
    style_even = '<span class=even>%s</span>'
    style_odd = '<span class=odd>%s</span>'
    style_left = '<span class=left>%s</span>'
    style_right = '<span class=right>%s</span>'

class HTMLTheme2(HTMLTheme):
    style_prompt = '#[#span class=prompt#]#%s#[#/span#]#'
    style_not_printable = '#[#span class=not_printable#]#%s#[#/span#]#'
    style_layer_name = '#[#span class=layer_name#]#%s#[#/span#]#'
    style_field_name = '#[#span class=field_name#]#%s#[#/span#]#'
    style_field_value = '#[#span class=field_value#]#%s#[#/span#]#'
    style_emph_field_name = '#[#span class=emph_field_name#]#%s#[#/span#]#'
    style_emph_field_value = '#[#span class=emph_field_value#]#%s#[#/span#]#'
    style_packetlist_name = '#[#span class=packetlist_name#]#%s#[#/span#]#'
    style_packetlist_proto = '#[#span class=packetlist_proto#]#%s#[#/span#]#'
    style_packetlist_value = '#[#span class=packetlist_value#]#%s#[#/span#]#'
    style_fail = '#[#span class=fail#]#%s#[#/span#]#'
    style_success = '#[#span class=success#]#%s#[#/span#]#'
    style_even = '#[#span class=even#]#%s#[#/span#]#'
    style_odd = '#[#span class=odd#]#%s#[#/span#]#'
    style_left = '#[#span class=left#]#%s#[#/span#]#'
    style_right = '#[#span class=right#]#%s#[#/span#]#'

def apply_ipython_style(shell):
    if False:
        return 10
    'Updates the specified IPython console shell with\n    the conf.color_theme scapy theme.'
    try:
        from IPython.terminal.prompts import Prompts, Token
    except Exception:
        from scapy.error import log_loading
        log_loading.warning("IPython too old. Shell color won't be handled.")
        return
    from scapy.config import conf
    scapy_style = {}
    if isinstance(conf.color_theme, NoTheme):
        shell.colors = 'nocolor'
    elif isinstance(conf.color_theme, BrightTheme):
        shell.colors = 'lightbg'
    elif isinstance(conf.color_theme, ColorOnBlackTheme):
        shell.colors = 'linux'
    else:
        shell.colors = 'neutral'
    try:
        get_ipython()
        color_magic = shell.magics_manager.magics['line']['colors']
        color_magic(shell.colors)
    except NameError:
        pass
    if isinstance(conf.prompt, Prompts):
        shell.prompts_class = conf.prompt
    else:
        if isinstance(conf.color_theme, (FormatTheme, NoTheme)):
            if isinstance(conf.color_theme, HTMLTheme):
                prompt = html.escape(conf.prompt)
            elif isinstance(conf.color_theme, LatexTheme):
                from scapy.utils import tex_escape
                prompt = tex_escape(conf.prompt)
            else:
                prompt = conf.prompt
            prompt = conf.color_theme.prompt(prompt)
        else:
            prompt = str(conf.prompt)
            scapy_style[Token.Prompt] = Color.ansi_to_pygments(conf.color_theme.style_prompt)

        class ClassicPrompt(Prompts):

            def in_prompt_tokens(self, cli=None):
                if False:
                    print('Hello World!')
                return [(Token.Prompt, prompt)]

            def out_prompt_tokens(self):
                if False:
                    return 10
                return [(Token.OutPrompt, '')]
        shell.prompts_class = ClassicPrompt
        sys.ps1 = prompt
    shell.highlighting_style_overrides = scapy_style
    try:
        get_ipython().refresh_style()
    except NameError:
        pass