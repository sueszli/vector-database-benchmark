"""Update pygments style

Call this script after each upgrade of pygments

"""
from pathlib import Path
import pygments
from pygments.formatters import HtmlFormatter
from searx import searx_dir
LESS_FILE = Path(searx_dir) / 'static/themes/simple/src/generated/pygments.less'
HEADER = f'/*\n   this file is generated automatically by searxng_extra/update/update_pygments.py\n   using pygments version {pygments.__version__}\n*/\n\n'
START_LIGHT_THEME = '\n.code-highlight {\n'
END_LIGHT_THEME = '\n}\n'
START_DARK_THEME = '\n.code-highlight-dark(){\n  .code-highlight {\n'
END_DARK_THEME = '\n  }\n}\n'

class Formatter(HtmlFormatter):

    @property
    def _pre_style(self):
        if False:
            print('Hello World!')
        return 'line-height: 100%;'

    def get_style_lines(self, arg=None):
        if False:
            return 10
        style_lines = []
        style_lines.extend(self.get_linenos_style_defs())
        style_lines.extend(self.get_background_style_defs(arg))
        style_lines.extend(self.get_token_style_defs(arg))
        return style_lines

def generat_css(light_style, dark_style) -> str:
    if False:
        print('Hello World!')
    css = HEADER + START_LIGHT_THEME
    for line in Formatter(style=light_style).get_style_lines():
        css += '\n  ' + line
    css += END_LIGHT_THEME + START_DARK_THEME
    for line in Formatter(style=dark_style).get_style_lines():
        css += '\n    ' + line
    css += END_DARK_THEME
    return css
if __name__ == '__main__':
    print('update: %s' % LESS_FILE)
    with open(LESS_FILE, 'w') as f:
        f.write(generat_css('default', 'lightbulb'))