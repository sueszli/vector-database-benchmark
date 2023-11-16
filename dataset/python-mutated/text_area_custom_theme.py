from rich.style import Style
from textual._text_area_theme import TextAreaTheme
from textual.app import App, ComposeResult
from textual.widgets import TextArea
TEXT = '# says hello\ndef hello(name):\n    print("hello" + name)\n\n# says goodbye\ndef goodbye(name):\n    print("goodbye" + name)\n'
MY_THEME = TextAreaTheme(name='my_cool_theme', cursor_style=Style(color='white', bgcolor='blue'), cursor_line_style=Style(bgcolor='yellow'), syntax_styles={'string': Style(color='red'), 'comment': Style(color='magenta')})

class TextAreaCustomThemes(App):

    def compose(self) -> ComposeResult:
        if False:
            return 10
        text_area = TextArea(TEXT, language='python')
        text_area.cursor_blink = False
        text_area.register_theme(MY_THEME)
        text_area.theme = 'my_cool_theme'
        yield text_area
app = TextAreaCustomThemes()
if __name__ == '__main__':
    app.run()