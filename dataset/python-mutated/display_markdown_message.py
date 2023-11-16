from rich import print as rich_print
from rich.markdown import Markdown
from rich.rule import Rule

def display_markdown_message(message):
    if False:
        return 10
    '\n    Display markdown message. Works with multiline strings with lots of indentation.\n    Will automatically make single line > tags beautiful.\n    '
    for line in message.split('\n'):
        line = line.strip()
        if line == '':
            print('')
        elif line == '---':
            rich_print(Rule(style='white'))
        else:
            rich_print(Markdown(line))
    if '\n' not in message and message.startswith('>'):
        print('')