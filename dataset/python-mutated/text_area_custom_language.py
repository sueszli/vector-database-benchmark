from pathlib import Path
from tree_sitter_languages import get_language
from textual.app import App, ComposeResult
from textual.widgets import TextArea
java_language = get_language('java')
java_highlight_query = (Path(__file__).parent / 'java_highlights.scm').read_text()
java_code = 'class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println("Hello, World!");\n    }\n}\n'

class TextAreaCustomLanguage(App):

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        text_area = TextArea(text=java_code)
        text_area.cursor_blink = False
        text_area.register_language(java_language, java_highlight_query)
        text_area.language = 'java'
        yield text_area
app = TextAreaCustomLanguage()
if __name__ == '__main__':
    app.run()