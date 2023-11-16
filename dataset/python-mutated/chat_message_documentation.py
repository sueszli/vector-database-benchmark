from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        while True:
            i = 10
    ui.chat_message('Hello NiceGUI!', name='Robot', stamp='now', avatar='https://robohash.org/ui')

def more() -> None:
    if False:
        i = 10
        return i + 15

    @text_demo('HTML text', '\n        Using the `text_html` parameter, you can send HTML text to the chat.\n    ')
    def html_text():
        if False:
            return 10
        ui.chat_message('Without <strong>HTML</strong>')
        ui.chat_message('With <strong>HTML</strong>', text_html=True)

    @text_demo('Newline', '\n        You can use newlines in the chat message.\n    ')
    def newline():
        if False:
            return 10
        ui.chat_message('This is a\nlong line!')

    @text_demo('Multi-part messages', '\n        You can send multiple message parts by passing a list of strings.\n    ')
    def multiple_messages():
        if False:
            print('Hello World!')
        ui.chat_message(['Hi! 😀', 'How are you?'])