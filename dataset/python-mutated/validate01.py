from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Button, RichLog

class ValidateApp(App):
    CSS_PATH = 'validate01.tcss'
    count = reactive(0)

    def validate_count(self, count: int) -> int:
        if False:
            while True:
                i = 10
        'Validate value.'
        if count < 0:
            count = 0
        elif count > 10:
            count = 10
        return count

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        yield Horizontal(Button('+1', id='plus', variant='success'), Button('-1', id='minus', variant='error'), id='buttons')
        yield RichLog(highlight=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if False:
            for i in range(10):
                print('nop')
        if event.button.id == 'plus':
            self.count += 1
        else:
            self.count -= 1
        self.query_one(RichLog).write(f'count = {self.count}')
if __name__ == '__main__':
    app = ValidateApp()
    app.run()