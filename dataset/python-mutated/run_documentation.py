from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        while True:
            i = 10
    ui.label('page with custom title')
main_demo.tab = 'My App'

def more() -> None:
    if False:
        while True:
            i = 10

    @text_demo('Emoji favicon', '\n        You can use an emoji as favicon.\n        This works in Chrome, Firefox and Safari.\n    ', tab=lambda : ui.markdown('🚀&nbsp; NiceGUI'))
    def emoji_favicon():
        if False:
            i = 10
            return i + 15
        ui.label('NiceGUI Rocks!')

    @text_demo('Base64 favicon', '\n        You can also use an base64-encoded image as favicon.\n    ', tab=lambda : (ui.image('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==').classes('w-4 h-4'), ui.label('NiceGUI')))
    def base64_favicon():
        if False:
            for i in range(10):
                print('nop')
        ui.label('NiceGUI with a red dot!')
        icon = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=='

    @text_demo('SVG favicon', '\n        And directly use an SVG as favicon.\n        Works in Chrome, Firefox and Safari.\n    ', tab=lambda : (ui.html('\n            <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">\n                <circle cx="100" cy="100" r="78" fill="#ffde34" stroke="black" stroke-width="3" />\n                <circle cx="80" cy="85" r="8" />\n                <circle cx="120" cy="85" r="8" />\n                <path d="m60,120 C75,150 125,150 140,120" style="fill:none; stroke:black; stroke-width:8; stroke-linecap:round" />\n            </svg>\n        ').classes('w-4 h-4'), ui.label('NiceGUI')))
    def svg_favicon():
        if False:
            i = 10
            return i + 15
        ui.label('NiceGUI makes you smile!')
        smiley = '\n            <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">\n                <circle cx="100" cy="100" r="78" fill="#ffde34" stroke="black" stroke-width="3" />\n                <circle cx="80" cy="85" r="8" />\n                <circle cx="120" cy="85" r="8" />\n                <path d="m60,120 C75,150 125,150 140,120" style="fill:none; stroke:black; stroke-width:8; stroke-linecap:round" />\n            </svg>\n        '