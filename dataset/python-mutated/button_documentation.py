from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        return 10
    ui.button('Click me!', on_click=lambda : ui.notify('You clicked me!'))

def more() -> None:
    if False:
        print('Hello World!')

    @text_demo('Icons', '\n        You can also add an icon to a button.\n    ')
    def icons() -> None:
        if False:
            while True:
                i = 10
        with ui.row():
            ui.button('demo', icon='history')
            ui.button(icon='thumb_up')
            with ui.button():
                ui.label('sub-elements')
                ui.image('https://picsum.photos/id/377/640/360').classes('rounded-full w-16 h-16 ml-4')

    @text_demo('Await button click', '\n        Sometimes it is convenient to wait for a button click before continuing the execution.\n    ')
    async def await_button_click() -> None:
        b = ui.button('Step')
        await b.clicked()
        ui.label('One')
        await b.clicked()
        ui.label('Two')
        await b.clicked()
        ui.label('Three')

    @text_demo('Disable button with a context manager', '\n        This showcases a context manager that can be used to disable a button for the duration of an async process.\n    ')
    def disable_context_manager() -> None:
        if False:
            print('Hello World!')
        from contextlib import contextmanager
        import httpx

        @contextmanager
        def disable(button: ui.button) -> None:
            if False:
                while True:
                    i = 10
            button.disable()
            try:
                yield
            finally:
                button.enable()

        async def get_slow_response(button: ui.button) -> None:
            with disable(button):
                async with httpx.AsyncClient() as client:
                    response = await client.get('https://httpbin.org/delay/1', timeout=5)
                    ui.notify(f'Response code: {response.status_code}')
        ui.button('Get slow response', on_click=lambda e: get_slow_response(e.sender))