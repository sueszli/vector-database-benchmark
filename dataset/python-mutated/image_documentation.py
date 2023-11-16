from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        i = 10
        return i + 15
    ui.image('https://picsum.photos/id/377/640/360')

def more() -> None:
    if False:
        return 10
    ui.add_body_html('<script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>')

    @text_demo('Local files', '\n        You can use local images as well by passing a path to the image file.\n    ')
    def local():
        if False:
            print('Hello World!')
        ui.image('website/static/logo.png').classes('w-16')

    @text_demo('Base64 string', '\n        You can also use a Base64 string as image source.\n    ')
    def base64():
        if False:
            print('Hello World!')
        base64 = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=='
        ui.image(base64).classes('w-2 h-2 m-auto')

    @text_demo('Lottie files', '\n        You can also use [Lottie files](https://lottiefiles.com/) with animations.\n    ')
    def lottie():
        if False:
            i = 10
            return i + 15
        src = 'https://assets1.lottiefiles.com/datafiles/HN7OcWNnoqje6iXIiZdWzKxvLIbfeCGTmvXmEm1h/data.json'
        ui.html(f'<lottie-player src="{src}" loop autoplay />').classes('w-full')

    @text_demo('Image link', '\n        Images can link to another page by wrapping them in a [ui.link](https://nicegui.io/documentation/link).\n    ')
    def link():
        if False:
            for i in range(10):
                print('nop')
        with ui.link(target='https://github.com/zauberzeug/nicegui'):
            ui.image('https://picsum.photos/id/41/640/360').classes('w-64')

    @text_demo('Force reload', '\n        You can force an image to reload by calling the `force_reload` method.\n        It will append a timestamp to the image URL, which will make the browser reload the image.\n    ')
    def force_reload():
        if False:
            for i in range(10):
                print('nop')
        img = ui.image('https://picsum.photos/640/360').classes('w-64')
        ui.button('Force reload', on_click=img.force_reload)