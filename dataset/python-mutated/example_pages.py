import theme
from message import message
from nicegui import ui

def create() -> None:
    if False:
        for i in range(10):
            print('nop')

    @ui.page('/a')
    def example_page_a():
        if False:
            for i in range(10):
                print('nop')
        with theme.frame('- Example A -'):
            message('Example A')

    @ui.page('/b')
    def example_page_b():
        if False:
            return 10
        with theme.frame('- Example B -'):
            message('Example B')