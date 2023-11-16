from nicegui import ui

from .screen import Screen


def test_splitter(screen: Screen):
    with ui.splitter() as splitter:
        with splitter.before:
            ui.label('Left hand side.')
        with splitter.after:
            ui.label('Right hand side.')
    ui.label().bind_text_from(splitter, 'value')

    screen.open('/')
    screen.should_contain('Left hand side.')
    screen.should_contain('Right hand side.')
    screen.should_contain('50')
    # TODO: programmatically move splitter
