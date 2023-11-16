from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        while True:
            i = 10
    ui.button('Say hi!', on_click=lambda : ui.notify('Hi!', close_button='OK'))

def more() -> None:
    if False:
        return 10

    @text_demo('Notification Types', '\n        There are different types that can be used to indicate the nature of the notification.\n    ')
    def notify_colors():
        if False:
            print('Hello World!')
        ui.button('negative', on_click=lambda : ui.notify('error', type='negative'))
        ui.button('positive', on_click=lambda : ui.notify('success', type='positive'))
        ui.button('warning', on_click=lambda : ui.notify('warning', type='warning'))

    @text_demo('Multiline Notifications', '\n        To allow a notification text to span multiple lines, it is sufficient to set `multi_line=True`.\n        If manual newline breaks are required (e.g. `\n`), you need to define a CSS style and pass it to the notification as shown in the example.\n    ')
    def multiline():
        if False:
            for i in range(10):
                print('nop')
        ui.html('<style>.multi-line-notification { white-space: pre-line; }</style>')
        ui.button('show', on_click=lambda : ui.notify('Lorem ipsum dolor sit amet, consectetur adipisicing elit. \nHic quisquam non ad sit assumenda consequuntur esse inventore officia. \nCorrupti reiciendis impedit vel, fugit odit quisquam quae porro exercitationem eveniet quasi.', multi_line=True, classes='multi-line-notification'))