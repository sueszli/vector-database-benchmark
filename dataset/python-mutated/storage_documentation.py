from collections import Counter
from datetime import datetime
from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        return 10
    "Storage\n\n    NiceGUI offers a straightforward method for data persistence within your application. \n    It features three built-in storage types:\n\n    - `app.storage.user`:\n        Stored server-side, each dictionary is associated with a unique identifier held in a browser session cookie.\n        Unique to each user, this storage is accessible across all their browser tabs.\n        `app.storage.browser['id']` is used to identify the user.\n    - `app.storage.general`:\n        Also stored server-side, this dictionary provides a shared storage space accessible to all users.\n    - `app.storage.browser`:\n        Unlike the previous types, this dictionary is stored directly as the browser session cookie, shared among all browser tabs for the same user.\n        However, `app.storage.user` is generally preferred due to its advantages in reducing data payload, enhancing security, and offering larger storage capacity.\n        By default, NiceGUI holds a unique identifier for the browser session in `app.storage.browser['id']`.\n\n    The user storage and browser storage are only available within `page builder functions </documentation/page>`_\n    because they are accessing the underlying `Request` object from FastAPI.\n    Additionally these two types require the `storage_secret` parameter in`ui.run()` to encrypt the browser session cookie.\n    "
    from nicegui import app
    app.storage.user['count'] = app.storage.user.get('count', 0) + 1
    with ui.row():
        ui.label('your own page visits:')
        ui.label().bind_text_from(app.storage.user, 'count')
counter = Counter()
start = datetime.now().strftime('%H:%M, %d %B %Y')

def more() -> None:
    if False:
        print('Hello World!')

    @text_demo('Counting page visits', '\n        Here we are using the automatically available browser-stored session ID to count the number of unique page visits.\n    ')
    def page_visits():
        if False:
            print('Hello World!')
        from collections import Counter
        from datetime import datetime
        from nicegui import app
        counter[app.storage.browser['id']] += 1
        ui.label(f'{len(counter)} unique views ({sum(counter.values())} overall) since {start}')

    @text_demo('Storing UI state', '\n        Storage can also be used in combination with [`bindings`](/documentation/bindings).\n        Here we are storing the value of a textarea between visits.\n        The note is also shared between all tabs of the same user.\n    ')
    def ui_state():
        if False:
            print('Hello World!')
        from nicegui import app
        ui.textarea('This note is kept between visits').classes('w-full').bind_value(app.storage.user, 'note')