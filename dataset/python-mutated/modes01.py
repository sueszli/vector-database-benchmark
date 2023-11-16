from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Placeholder

class DashboardScreen(Screen):

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        yield Placeholder('Dashboard Screen')
        yield Footer()

class SettingsScreen(Screen):

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Placeholder('Settings Screen')
        yield Footer()

class HelpScreen(Screen):

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield Placeholder('Help Screen')
        yield Footer()

class ModesApp(App):
    BINDINGS = [('d', "switch_mode('dashboard')", 'Dashboard'), ('s', "switch_mode('settings')", 'Settings'), ('h', "switch_mode('help')", 'Help')]
    MODES = {'dashboard': DashboardScreen, 'settings': SettingsScreen, 'help': HelpScreen}

    def on_mount(self) -> None:
        if False:
            return 10
        self.switch_mode('dashboard')
if __name__ == '__main__':
    app = ModesApp()
    app.run()