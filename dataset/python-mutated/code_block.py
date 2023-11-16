from rich.box import MINIMAL
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from .base_block import BaseBlock

class CodeBlock(BaseBlock):
    """
    Code Blocks display code and outputs in different languages. You can also set the active_line!
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.type = 'code'
        self.language = ''
        self.output = ''
        self.code = ''
        self.active_line = None
        self.margin_top = True

    def refresh(self, cursor=True):
        if False:
            i = 10
            return i + 15
        code = self.code
        if not code:
            return
        code_table = Table(show_header=False, show_footer=False, box=None, padding=0, expand=True)
        code_table.add_column()
        if cursor:
            code += '●'
        code_lines = code.strip().split('\n')
        for (i, line) in enumerate(code_lines, start=1):
            if i == self.active_line:
                syntax = Syntax(line, self.language, theme='bw', line_numbers=False, word_wrap=True)
                code_table.add_row(syntax, style='black on white')
            else:
                syntax = Syntax(line, self.language, theme='monokai', line_numbers=False, word_wrap=True)
                code_table.add_row(syntax)
        code_panel = Panel(code_table, box=MINIMAL, style='on #272722')
        if self.output == '' or self.output == 'None':
            output_panel = ''
        else:
            output_panel = Panel(self.output, box=MINIMAL, style='#FFFFFF on #3b3b37')
        group_items = [code_panel, output_panel]
        if self.margin_top:
            group_items = [''] + group_items
        group = Group(*group_items)
        self.live.update(group)
        self.live.refresh()