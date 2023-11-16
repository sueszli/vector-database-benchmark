from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax

def process_syntax(code, lang, theme, line_numbers, code_width, word_wrap):
    if False:
        while True:
            i = 10
    syntax = Syntax(code, lang, theme=theme, line_numbers=line_numbers, code_width=code_width, word_wrap=word_wrap)
    return syntax

def get_code_without_tag(code, tag):
    if False:
        i = 10
        return i + 15
    tag_start = '<%s ' % tag
    tag_solo = '<%s>' % tag
    tag_end = '</%s>' % tag
    while tag_start in code:
        start = code.find(tag_start)
        if start == -1:
            break
        end = code.find('>', start + 1) + 1
        if end == 0:
            break
        code = code[:start] + code[end:]
    code = code.replace(tag_solo, '')
    code = code.replace(tag_end, '')
    return code

def display_markdown(code):
    if False:
        print('Hello World!')
    try:
        markdown = Markdown(code)
        console = Console()
        console.print(markdown)
        return True
    except Exception:
        return False

def display_code(code):
    if False:
        i = 10
        return i + 15
    try:
        console = Console()
        console.print(code)
        return True
    except Exception:
        return False

def fix_emoji_spacing(code):
    if False:
        while True:
            i = 10
    try:
        double_width_emojis = ['👁️', '🗺️', '🖼️', '🗄️', '♻️', '🗂️', '🖥️', '🕹️', '🎞️', '🎛️', '🎖️', '☀️', '⏺️', '▶️', '↘️', '⬇️', '↙️', '⬅️', '↖️', '⬆️', '↗️', '➡️']
        for emoji in double_width_emojis:
            if emoji in code:
                code = code.replace(emoji, emoji + ' ')
        code = code.replace('✅<', '✅ <')
        code = code.replace('❌<', '❌ <')
    except Exception:
        pass
    return code