from nicegui import ui
from ..documentation_tools import text_demo
from ..style import link_target

def main_demo() -> None:
    if False:
        while True:
            i = 10
    ui.link('NiceGUI on GitHub', 'https://github.com/zauberzeug/nicegui')

def more() -> None:
    if False:
        return 10

    @text_demo('Navigate on large pages', "\n        To jump to a specific location within a page you can place linkable anchors with `ui.link_target('target_name')`\n        or simply pass a NiceGUI element as link target.\n    ")
    def same_page_links():
        if False:
            for i in range(10):
                print('nop')
        navigation = ui.row()
        link_target('target_A', '-10px')
        ui.label('Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.')
        link_target('target_B', '70px')
        label_B = ui.label('Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.')
        with navigation:
            ui.link('Goto A', '#target_A')
            ui.link('Goto B', '#target_B')

    @text_demo('Links to other pages', '\n        You can link to other pages by providing the link target as path or function reference.\n    ')
    def link_to_other_page():
        if False:
            while True:
                i = 10

        @ui.page('/some_other_page')
        def my_page():
            if False:
                i = 10
                return i + 15
            ui.label('This is another page')
        ui.label('Go to other page')
        ui.link('... with path', '/some_other_page')
        ui.link('... with function reference', my_page)

    @text_demo('Link from images and other elements', '\n        By nesting elements inside a link you can make the whole element clickable.\n        This works with all elements but is most useful for non-interactive elements like \n        [ui.image](/documentation/image), [ui.avatar](/documentation/image) etc.\n    ')
    def link_from_elements():
        if False:
            for i in range(10):
                print('nop')
        with ui.link(target='https://github.com/zauberzeug/nicegui'):
            ui.image('https://picsum.photos/id/41/640/360').classes('w-64')