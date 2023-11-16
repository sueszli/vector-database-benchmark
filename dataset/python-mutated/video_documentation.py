from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        return 10
    v = ui.video('https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4')
    v.on('ended', lambda _: ui.notify('Video playback completed'))

def more() -> None:
    if False:
        while True:
            i = 10

    @text_demo('Control the video element', '\n        This demo shows how to play, pause and seek programmatically.\n    ')
    def control_demo() -> None:
        if False:
            while True:
                i = 10
        v = ui.video('https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4')
        ui.button('Play', on_click=v.play)
        ui.button('Pause', on_click=v.pause)
        ui.button('Jump to 0:05', on_click=lambda : v.seek(5))