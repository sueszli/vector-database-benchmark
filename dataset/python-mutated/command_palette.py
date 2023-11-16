from textual.app import App
from textual.command import Hit, Hits, Provider

class TestSource(Provider):

    def goes_nowhere_does_nothing(self) -> None:
        if False:
            return 10
        pass

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for n in range(10):
            command = f'This is a test of this code {n}'
            yield Hit(n / 10, matcher.highlight(command), self.goes_nowhere_does_nothing, command)

class CommandPaletteApp(App[None]):
    COMMANDS = {TestSource}

    def on_mount(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.action_command_palette()
if __name__ == '__main__':
    CommandPaletteApp().run()