from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        i = 10
        return i + 15
    ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')

def more() -> None:
    if False:
        print('Hello World!')

    @text_demo('Upload restrictions', '\n        In this demo, the upload is restricted to a maximum file size of 1 MB.\n        When a file is rejected, a notification is shown.\n    ')
    def upload_restrictions() -> None:
        if False:
            for i in range(10):
                print('nop')
        ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}'), on_rejected=lambda : ui.notify('Rejected!'), max_file_size=1000000).classes('max-w-full')

    @text_demo('Show file content', '\n        In this demo, the uploaded markdown file is shown in a dialog.\n    ')
    def show_file_content() -> None:
        if False:
            print('Hello World!')
        from nicegui import events
        with ui.dialog().props('full-width') as dialog:
            with ui.card():
                content = ui.markdown()

        def handle_upload(e: events.UploadEventArguments):
            if False:
                i = 10
                return i + 15
            text = e.content.read().decode('utf-8')
            content.set_content(text)
            dialog.open()
        ui.upload(on_upload=handle_upload).props('accept=.md').classes('max-w-full')