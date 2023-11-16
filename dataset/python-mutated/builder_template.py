from kivy.lang import Builder
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
Builder.load_string("\n[BlehItem@BoxLayout]:\n    orientation: 'vertical'\n    Label:\n        text: str(ctx.idx)\n    Button:\n        text: ctx.word\n")

class BlehApp(App):

    def build(self):
        if False:
            while True:
                i = 10
        root = BoxLayout()
        for (idx, word) in enumerate(('Hello', 'World')):
            wid = Builder.template('BlehItem', **{'idx': idx, 'word': word})
            root.add_widget(wid)
        return root
if __name__ == '__main__':
    BlehApp().run()