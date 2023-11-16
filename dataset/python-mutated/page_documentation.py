from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        print('Hello World!')

    @ui.page('/other_page')
    def other_page():
        if False:
            return 10
        ui.label('Welcome to the other side')
        ui.link('Back to main page', '/documentation#page')

    @ui.page('/dark_page', dark=True)
    def dark_page():
        if False:
            print('Hello World!')
        ui.label('Welcome to the dark side')
        ui.link('Back to main page', '/documentation#page')
    ui.link('Visit other page', other_page)
    ui.link('Visit dark page', dark_page)

def more() -> None:
    if False:
        i = 10
        return i + 15

    @text_demo('Pages with Path Parameters', '\n        Page routes can contain parameters like [FastAPI](https://fastapi.tiangolo.com/tutorial/path-params/>).\n        If type-annotated, they are automatically converted to bool, int, float and complex values.\n        If the page function expects a `request` argument, the request object is automatically provided.\n        The `client` argument provides access to the websocket connection, layout, etc.\n    ')
    def page_with_path_parameters_demo():
        if False:
            return 10

        @ui.page('/repeat/{word}/{count}')
        def page(word: str, count: int):
            if False:
                for i in range(10):
                    print('nop')
            ui.label(word * count)
        ui.link('Say hi to Santa!', '/repeat/Ho! /3')

    @text_demo('Wait for Client Connection', '\n        To wait for a client connection, you can add a `client` argument to the decorated page function\n        and await `client.connected()`.\n        All code below that statement is executed after the websocket connection between server and client has been established.\n\n        For example, this allows you to run JavaScript commands; which is only possible with a client connection (see [#112](https://github.com/zauberzeug/nicegui/issues/112)).\n        Also it is possible to do async stuff while the user already sees some content.\n    ')
    def wait_for_connected_demo():
        if False:
            i = 10
            return i + 15
        import asyncio
        from nicegui import Client

        @ui.page('/wait_for_connection')
        async def wait_for_connection(client: Client):
            ui.label('This text is displayed immediately.')
            await client.connected()
            await asyncio.sleep(2)
            ui.label('This text is displayed 2 seconds after the page has been fully loaded.')
            ui.label(f'The IP address {client.ip} was obtained from the websocket.')
        ui.link('wait for connection', wait_for_connection)

    @text_demo('Modularize with APIRouter', "\n        You can use the NiceGUI specialization of\n        [FastAPI's APIRouter](https://fastapi.tiangolo.com/tutorial/bigger-applications/?h=apirouter#apirouter)\n        to modularize your code by grouping pages and other routes together.\n        This is especially useful if you want to reuse the same prefix for multiple pages.\n        The router and its pages can be neatly tugged away in a separate module (e.g. file) and\n        the router is simply imported and included in the main app.\n        See our [modularization example](https://github.com/zauberzeug/nicegui/blob/main/examples/modularization/example_c.py)\n        for a multi-file app structure.\n    ", tab='/sub-path')
    def api_router_demo():
        if False:
            i = 10
            return i + 15
        ui.label('Shows up on /sub-path')