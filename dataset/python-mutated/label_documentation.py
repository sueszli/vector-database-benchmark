from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        while True:
            i = 10
    ui.label('some label')

def more() -> None:
    if False:
        return 10

    @text_demo('Change Appearance Depending on the Content', '\n        You can overwrite the `_handle_text_change` method to update other attributes of a label depending on its content. \n        This technique also works for bindings as shown in the example below.\n    ')
    def status():
        if False:
            i = 10
            return i + 15

        class status_label(ui.label):

            def _handle_text_change(self, text: str) -> None:
                if False:
                    i = 10
                    return i + 15
                super()._handle_text_change(text)
                if text == 'ok':
                    self.classes(replace='text-positive')
                else:
                    self.classes(replace='text-negative')
        model = {'status': 'error'}
        status_label().bind_text_from(model, 'status')
        ui.switch(on_change=lambda e: model.update(status='ok' if e.value else 'error'))