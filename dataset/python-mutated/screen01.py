from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Static
ERROR_TEXT = '\nAn error has occurred. To continue:\n\nPress Enter to return to Windows, or\n\nPress CTRL+ALT+DEL to restart your computer. If you do this,\nyou will lose any unsaved information in all open applications.\n\nError: 0E : 016F : BFF9B3D4\n'

class BSOD(Screen):
    BINDINGS = [('escape', 'app.pop_screen', 'Pop screen')]

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Static(' Windows ', id='title')
        yield Static(ERROR_TEXT)
        yield Static('Press any key to continue [blink]_[/]', id='any-key')

class BSODApp(App):
    CSS_PATH = 'screen01.tcss'
    SCREENS = {'bsod': BSOD()}
    BINDINGS = [('b', "push_screen('bsod')", 'BSOD')]
if __name__ == '__main__':
    app = BSODApp()
    app.run()