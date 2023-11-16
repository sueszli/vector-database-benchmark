from textual.app import App, ComposeResult
from textual.widgets import MarkdownViewer
EXAMPLE_MARKDOWN = '# Markdown Viewer\n\nThis is an example of Textual\'s `MarkdownViewer` widget.\n\n\n## Features\n\nMarkdown syntax and extensions are supported.\n\n- Typography *emphasis*, **strong**, `inline code` etc.\n- Headers\n- Lists (bullet and ordered)\n- Syntax highlighted code blocks\n- Tables!\n\n## Tables\n\nTables are displayed in a DataTable widget.\n\n| Name            | Type   | Default | Description                        |\n| --------------- | ------ | ------- | ---------------------------------- |\n| `show_header`   | `bool` | `True`  | Show the table header              |\n| `fixed_rows`    | `int`  | `0`     | Number of fixed rows               |\n| `fixed_columns` | `int`  | `0`     | Number of fixed columns            |\n| `zebra_stripes` | `bool` | `False` | Display alternating colors on rows |\n| `header_height` | `int`  | `1`     | Height of header row               |\n| `show_cursor`   | `bool` | `True`  | Show a cell cursor                 |\n\n\n## Code Blocks\n\nCode blocks are syntax highlighted, with guidelines.\n\n```python\nclass ListViewExample(App):\n    def compose(self) -> ComposeResult:\n        yield ListView(\n            ListItem(Label("One")),\n            ListItem(Label("Two")),\n            ListItem(Label("Three")),\n        )\n        yield Footer()\n```\n'

class MarkdownExampleApp(App):

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield MarkdownViewer(EXAMPLE_MARKDOWN, show_table_of_contents=True)
if __name__ == '__main__':
    app = MarkdownExampleApp()
    app.run()