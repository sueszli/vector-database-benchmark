from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        while True:
            i = 10
    ui.markdown('This is **Markdown**.')

def more() -> None:
    if False:
        return 10

    @text_demo('Markdown with indentation', '\n        Common indentation is automatically stripped from the beginning of each line.\n        So you can indent markdown elements, and they will still be rendered correctly.\n    ')
    def markdown_with_indentation():
        if False:
            for i in range(10):
                print('nop')
        ui.markdown('\n            ### Example\n\n            This line is not indented.\n\n                This block is indented.\n                Thus it is rendered as source code.\n            \n            This is normal text again.\n        ')

    @text_demo('Markdown with code blocks', '\n        You can use code blocks to show code examples.\n        If you specify the language after the opening triple backticks, the code will be syntax highlighted.\n        See [the Pygments website](https://pygments.org/languages/) for a list of supported languages.\n    ')
    def markdown_with_code_blocks():
        if False:
            print('Hello World!')
        ui.markdown("\n            ```python\n            from nicegui import ui\n\n            ui.label('Hello World!')\n\n            ui.run(dark=True)\n            ```\n        ")

    @text_demo('Markdown tables', '\n        By activating the "tables" extra, you can use Markdown tables.\n        See the [markdown2 documentation](https://github.com/trentm/python-markdown2/wiki/Extras#implemented-extras) for a list of available extras.\n    ')
    def markdown_with_code_blocks():
        if False:
            for i in range(10):
                print('nop')
        ui.markdown('\n            | First name | Last name |\n            | ---------- | --------- |\n            | Max        | Planck    |\n            | Marie      | Curie     |\n        ', extras=['tables'])