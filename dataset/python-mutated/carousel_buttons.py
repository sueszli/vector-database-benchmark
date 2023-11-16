"""
Carousel example with button inside.
This is a tiny test for testing the scroll distance/timeout
And ensure the down/up are dispatched if no gesture is done.
"""
from kivy.uix.carousel import Carousel
from kivy.uix.gridlayout import GridLayout
from kivy.app import App
from kivy.lang import Builder
Builder.load_string("\n<Page>:\n    cols: 3\n    Label:\n        text: str(id(root))\n    Button\n    Button\n    Button\n    Button\n        text: 'load(page 3)'\n        on_release:\n            carousel = root.parent.parent\n            carousel.load_slide(carousel.slides[2])\n    Button\n    Button\n        text: 'prev'\n        on_release:\n            root.parent.parent.load_previous()\n    Button\n    Button\n        text: 'next'\n        on_release:\n            root.parent.parent.load_next()\n")

class Page(GridLayout):
    pass

class TestApp(App):

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        root = Carousel()
        for x in range(10):
            root.add_widget(Page())
        return root
if __name__ == '__main__':
    TestApp().run()