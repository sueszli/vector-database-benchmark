from textual.app import App
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Header, Label

class LabeledBox(Container):
    DEFAULT_CSS = '\n    LabeledBox {\n        layers: base_ top_;\n        width: 100%;\n        height: 100%;\n    }\n\n    LabeledBox > Container {\n        layer: base_;\n        border: round $primary;\n        width: 100%;\n        height: 100%;\n        layout: vertical;\n    }\n\n    LabeledBox > Label {\n        layer: top_;\n        offset-x: 2;\n    }\n    '

    def __init__(self, title, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.__label = Label(title)
        super().__init__(self.__label, Container(*args, **kwargs))

    @property
    def label(self):
        if False:
            i = 10
            return i + 15
        return self.__label

class StatusTable(DataTable):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.cursor_type = 'row'
        self.show_cursor = False
        self.add_column('Foo')
        self.add_column('Bar')
        self.add_column('Baz')
        for _ in range(50):
            self.add_row('ABCDEFGH', '0123456789', 'IJKLMNOPQRSTUVWXYZ')

class Status(LabeledBox):
    DEFAULT_CSS = '\n    Status {\n        width: auto;\n    }\n\n    Status Container {\n        width: auto;\n    }\n\n    Status StatusTable {\n        width: auto;\n        height: 100%;\n        margin-top: 1;\n        scrollbar-gutter: stable;\n        overflow-x: hidden;\n    }\n    '

    def __init__(self, name: str):
        if False:
            while True:
                i = 10
        self.__name = name
        self.__table = StatusTable()
        super().__init__(f' {self.__name} ', self.__table)

    @property
    def name(self) -> str:
        if False:
            return 10
        return self.__name

    @property
    def table(self) -> StatusTable:
        if False:
            while True:
                i = 10
        return self.__table

class Rendering(LabeledBox):
    DEFAULT_CSS = '\n    #issue-info {\n        height: auto;\n        border-bottom: dashed #632CA6;\n    }\n\n    #statuses-box {\n        height: 1fr;\n        width: auto;\n    }\n    '

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__info = Label('test')
        super().__init__('', ScrollableContainer(Horizontal(self.__info, id='issue-info'), Horizontal(*[Status(str(i)) for i in range(4)], id='statuses-box'), id='issues-box'))

    @property
    def info(self) -> Label:
        if False:
            return 10
        return self.__info

class Sidebar(LabeledBox):
    DEFAULT_CSS = '\n    #sidebar-status {\n        height: auto;\n        border-bottom: dashed #632CA6;\n    }\n\n    #sidebar-options {\n        height: 1fr;\n    }\n    '

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__status = Label('ok')
        self.__options = Vertical()
        super().__init__('', Container(self.__status, id='sidebar-status'), Container(self.__options, id='sidebar-options'))

    @property
    def status(self) -> Label:
        if False:
            return 10
        return self.__status

    @property
    def options(self) -> Vertical:
        if False:
            print('Hello World!')
        return self.__options

class MyScreen(Screen):
    DEFAULT_CSS = '\n    #main-content {\n        layout: grid;\n        grid-size: 2;\n        grid-columns: 1fr 5fr;\n        grid-rows: 1fr;\n    }\n\n    #main-content-sidebar {\n        height: 100%;\n    }\n\n    #main-content-rendering {\n        height: 100%;\n    }\n    '

    def compose(self):
        if False:
            i = 10
            return i + 15
        yield Header()
        yield Container(Container(Sidebar(), id='main-content-sidebar'), Container(Rendering(), id='main-content-rendering'), id='main-content')

class MyApp(App):

    async def on_mount(self):
        self.install_screen(MyScreen(), 'myscreen')
        await self.push_screen('myscreen')
if __name__ == '__main__':
    app = MyApp()
    app.run()