from textual.app import App, ComposeResult
from textual.widgets import Footer, Label, Markdown, TabbedContent, TabPane
LETO = '\n# Duke Leto I Atreides\n\nHead of House Atreides.\n'
JESSICA = '\n# Lady Jessica\n\nBene Gesserit and concubine of Leto, and mother of Paul and Alia.\n'
PAUL = '\n# Paul Atreides\n\nSon of Leto and Jessica.\n'

class TabbedApp(App):
    """An example of tabbed content."""
    BINDINGS = [('l', "show_tab('leto')", 'Leto'), ('j', "show_tab('jessica')", 'Jessica'), ('p', "show_tab('paul')", 'Paul')]

    def compose(self) -> ComposeResult:
        if False:
            return 10
        'Compose app with tabbed content.'
        yield Footer()
        with TabbedContent(initial='jessica'):
            with TabPane('Leto', id='leto'):
                yield Markdown(LETO)
            with TabPane('Jessica', id='jessica'):
                yield Markdown(JESSICA)
                with TabbedContent('Paul', 'Alia'):
                    yield TabPane('Paul', Label('First child'))
                    yield TabPane('Alia', Label('Second child'))
            with TabPane('Paul', id='paul'):
                yield Markdown(PAUL)

    def action_show_tab(self, tab: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Switch to a new tab.'
        self.get_child_by_type(TabbedContent).active = tab
if __name__ == '__main__':
    app = TabbedApp()
    app.run()