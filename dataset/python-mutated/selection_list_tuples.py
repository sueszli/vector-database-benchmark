from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, SelectionList

class SelectionListApp(App[None]):
    CSS_PATH = 'selection_list.tcss'

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Header()
        yield SelectionList[int](("Falken's Maze", 0, True), ('Black Jack', 1), ('Gin Rummy', 2), ('Hearts', 3), ('Bridge', 4), ('Checkers', 5), ('Chess', 6, True), ('Poker', 7), ('Fighter Combat', 8, True))
        yield Footer()

    def on_mount(self) -> None:
        if False:
            print('Hello World!')
        self.query_one(SelectionList).border_title = 'Shall we play some games?'
if __name__ == '__main__':
    SelectionListApp().run()