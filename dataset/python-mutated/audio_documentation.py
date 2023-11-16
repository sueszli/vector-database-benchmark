from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        for i in range(10):
            print('nop')
    a = ui.audio('https://cdn.pixabay.com/download/audio/2022/02/22/audio_d1718ab41b.mp3')
    a.on('ended', lambda _: ui.notify('Audio playback completed'))
    ui.button(on_click=lambda : a.props('muted'), icon='volume_off').props('outline')
    ui.button(on_click=lambda : a.props(remove='muted'), icon='volume_up').props('outline')

def more() -> None:
    if False:
        i = 10
        return i + 15

    @text_demo('Control the audio element', '\n        This demo shows how to play, pause and seek programmatically.\n    ')
    def control_demo() -> None:
        if False:
            for i in range(10):
                print('nop')
        a = ui.audio('https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3')
        ui.button('Play', on_click=a.play)
        ui.button('Pause', on_click=a.pause)
        ui.button('Jump to 0:30', on_click=lambda : a.seek(30))