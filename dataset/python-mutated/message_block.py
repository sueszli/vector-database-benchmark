import re
from rich.box import MINIMAL
from rich.markdown import Markdown
from rich.panel import Panel
from .base_block import BaseBlock

class MessageBlock(BaseBlock):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.type = 'message'
        self.message = ''
        self.has_run = False

    def refresh(self, cursor=True):
        if False:
            i = 10
            return i + 15
        content = textify_markdown_code_blocks(self.message)
        if cursor:
            content += '●'
        markdown = Markdown(content.strip())
        panel = Panel(markdown, box=MINIMAL)
        self.live.update(panel)
        self.live.refresh()

def textify_markdown_code_blocks(text):
    if False:
        for i in range(10):
            print('nop')
    "\n    To distinguish CodeBlocks from markdown code, we simply turn all markdown code\n    (like '```python...') into text code blocks ('```text') which makes the code black and white.\n    "
    replacement = '```text'
    lines = text.split('\n')
    inside_code_block = False
    for i in range(len(lines)):
        if re.match('^```(\\w*)$', lines[i].strip()):
            inside_code_block = not inside_code_block
            if inside_code_block:
                lines[i] = replacement
    return '\n'.join(lines)