from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, ContentSwitcher, DataTable, Markdown
MARKDOWN_EXAMPLE = "# Three Flavours Cornetto\n\nThe Three Flavours Cornetto trilogy is an anthology series of British\ncomedic genre films directed by Edgar Wright.\n\n## Shaun of the Dead\n\n| Flavour | UK Release Date | Director |\n| -- | -- | -- |\n| Strawberry | 2004-04-09 | Edgar Wright |\n\n## Hot Fuzz\n\n| Flavour | UK Release Date | Director |\n| -- | -- | -- |\n| Classico | 2007-02-17 | Edgar Wright |\n\n## The World's End\n\n| Flavour | UK Release Date | Director |\n| -- | -- | -- |\n| Mint | 2013-07-19 | Edgar Wright |\n"

class ContentSwitcherApp(App[None]):
    CSS_PATH = 'content_switcher.tcss'

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        with Horizontal(id='buttons'):
            yield Button('DataTable', id='data-table')
            yield Button('Markdown', id='markdown')
        with ContentSwitcher(initial='data-table'):
            yield DataTable(id='data-table')
            with VerticalScroll(id='markdown'):
                yield Markdown(MARKDOWN_EXAMPLE)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if False:
            i = 10
            return i + 15
        self.query_one(ContentSwitcher).current = event.button.id

    def on_mount(self) -> None:
        if False:
            while True:
                i = 10
        table = self.query_one(DataTable)
        table.add_columns('Book', 'Year')
        table.add_rows([(title.ljust(35), year) for (title, year) in (('Dune', 1965), ('Dune Messiah', 1969), ('Children of Dune', 1976), ('God Emperor of Dune', 1981), ('Heretics of Dune', 1984), ('Chapterhouse: Dune', 1985))])
if __name__ == '__main__':
    ContentSwitcherApp().run()