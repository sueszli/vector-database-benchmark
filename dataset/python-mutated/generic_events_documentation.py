from nicegui import context, ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        while True:
            i = 10
    'Generic Events\n\n    Most UI elements come with predefined events.\n    For example, a `ui.button` like "A" in the demo has an `on_click` parameter that expects a coroutine or function.\n    But you can also use the `on` method to register a generic event handler like for "B".\n    This allows you to register handlers for any event that is supported by JavaScript and Quasar.\n\n    For example, you can register a handler for the `mousemove` event like for "C", even though there is no `on_mousemove` parameter for `ui.button`.\n    Some events, like `mousemove`, are fired very often.\n    To avoid performance issues, you can use the `throttle` parameter to only call the handler every `throttle` seconds ("D").\n\n    The generic event handler can be synchronous or asynchronous and optionally takes `GenericEventArguments` as argument ("E").\n    You can also specify which attributes of the JavaScript or Quasar event should be passed to the handler ("F").\n    This can reduce the amount of data that needs to be transferred between the server and the client.\n\n    Here you can find more information about the events that are supported:\n\n    - https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement#events for HTML elements\n    - https://quasar.dev/vue-components for Quasar-based elements (see the "Events" tab on the individual component page)\n    '
    with ui.row():
        ui.button('A', on_click=lambda : ui.notify('You clicked the button A.'))
        ui.button('B').on('click', lambda : ui.notify('You clicked the button B.'))
    with ui.row():
        ui.button('C').on('mousemove', lambda : ui.notify('You moved on button C.'))
        ui.button('D').on('mousemove', lambda : ui.notify('You moved on button D.'), throttle=0.5)
    with ui.row():
        ui.button('E').on('mousedown', lambda e: ui.notify(e))
        ui.button('F').on('mousedown', lambda e: ui.notify(e), ['ctrlKey', 'shiftKey'])

def more() -> None:
    if False:
        return 10

    @text_demo('Specifying event attributes', '\n        **A list of strings** names the attributes of the JavaScript event object:\n            ```py\n            ui.button().on(\'click\', handle_click, [\'clientX\', \'clientY\'])\n            ```\n\n        **An empty list** requests _no_ attributes:\n            ```py\n            ui.button().on(\'click\', handle_click, [])\n            ```\n\n        **The value `None`** represents _all_ attributes (the default):\n            ```py\n            ui.button().on(\'click\', handle_click, None)\n            ```\n\n        **If the event is called with multiple arguments** like QTable\'s "row-click" `(evt, row, index) => void`,\n            you can define a list of argument definitions:\n            ```py\n            ui.table(...).on(\'rowClick\', handle_click, [[], [\'name\'], None])\n            ```\n            In this example the "row-click" event will omit all arguments of the first `evt` argument,\n            send only the "name" attribute of the `row` argument and send the full `index`.\n\n        If the retrieved list of event arguments has length 1, the argument is automatically unpacked.\n        So you can write\n        ```py\n        ui.button().on(\'click\', lambda e: print(e.args[\'clientX\'], flush=True))\n        ```\n        instead of\n        ```py\n        ui.button().on(\'click\', lambda e: print(e.args[0][\'clientX\'], flush=True))\n        ```\n\n        Note that by default all JSON-serializable attributes of all arguments are sent.\n        This is to simplify registering for new events and discovering their attributes.\n        If bandwidth is an issue, the arguments should be limited to what is actually needed on the server.\n    ')
    def event_attributes() -> None:
        if False:
            while True:
                i = 10
        columns = [{'name': 'name', 'label': 'Name', 'field': 'name'}, {'name': 'age', 'label': 'Age', 'field': 'age'}]
        rows = [{'name': 'Alice', 'age': 42}, {'name': 'Bob', 'age': 23}]
        ui.table(columns, rows, 'name').on('rowClick', ui.notify, [[], ['name'], None])

    @text_demo('Modifiers', '\n        You can also include [key modifiers](https://vuejs.org/guide/essentials/event-handling.html#key-modifiers>) (shown in input "A"),\n        modifier combinations (shown in input "B"),\n        and [event modifiers](https://vuejs.org/guide/essentials/event-handling.html#mouse-button-modifiers>) (shown in input "C").\n    ')
    def modifiers() -> None:
        if False:
            print('Hello World!')
        with ui.row():
            ui.input('A').classes('w-12').on('keydown.space', lambda : ui.notify('You pressed space.'))
            ui.input('B').classes('w-12').on('keydown.y.shift', lambda : ui.notify('You pressed Shift+Y'))
            ui.input('C').classes('w-12').on('keydown.once', lambda : ui.notify('You started typing.'))

    @text_demo('Custom events', '\n        It is fairly easy to emit custom events from JavaScript which can be listened to with `element.on(...)`.\n        This can be useful if you want to call Python code when something happens in JavaScript.\n        In this example we are listening to the `visibilitychange` event of the browser tab.\n    ')
    async def custom_events() -> None:
        tabwatch = ui.checkbox('Watch browser tab re-entering').on('tabvisible', lambda : ui.notify('Welcome back!') if tabwatch.value else None, args=[])
        ui.add_head_html(f"\n            <script>\n            document.addEventListener('visibilitychange', () => {{\n                if (document.visibilityState === 'visible')\n                    getElement({tabwatch.id}).$emit('tabvisible');\n            }});\n            </script>\n        ")
        await context.get_client().connected()
        ui.run_javascript(f"\n            document.addEventListener('visibilitychange', () => {{\n                if (document.visibilityState === 'visible')\n                    getElement({tabwatch.id}).$emit('tabvisible');\n            }});\n        ")