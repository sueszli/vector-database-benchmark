from kivy.app import App
from kivy.lang import Builder
root = Builder.load_string("\nLabel:\n    text:\n        ('[b]Hello[/b] [color=ff0099]World[/color]\\n'\n        '[color=ff0099]Hello[/color] [b]World[/b]\\n'\n        '[b]Hello[/b] [color=ff0099]World[/color]')\n    markup: True\n    font_size: '64pt'\n")

class LabelWithMarkup(App):

    def build(self):
        if False:
            return 10
        return root
if __name__ == '__main__':
    LabelWithMarkup().run()