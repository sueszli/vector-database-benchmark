from textual.app import App
from textual.widgets import Label

class LinkHoverStyleApp(App):
    CSS_PATH = 'link_hover_style.tcss'

    def compose(self):
        if False:
            return 10
        yield Label('Visit the [link=https://textualize.io]Textualize[/link] website.', id='lbl1')
        yield Label('Click [@click=app.bell]here[/] for the bell sound.', id='lbl2')
        yield Label('You can also click [@click=app.bell]here[/] for the bell sound.', id='lbl3')
        yield Label('[@click=app.quit]Exit this application.[/]', id='lbl4')
if __name__ == '__main__':
    app = LinkHoverStyleApp()
    app.run()