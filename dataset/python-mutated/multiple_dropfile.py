from kivy.app import App
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout

class DropFile(Button):

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(DropFile, self).__init__(**kwargs)
        app = App.get_running_app()
        app.drops.append(self.on_drop_file)

    def on_drop_file(self, widget, filename):
        if False:
            return 10
        if self.collide_point(*Window.mouse_pos):
            self.text = filename.decode('utf-8')

class DropApp(App):

    def build(self):
        if False:
            i = 10
            return i + 15
        self.drops = []
        Window.bind(on_drop_file=self.handledrops)
        box = BoxLayout()
        dropleft = DropFile(text='left')
        box.add_widget(dropleft)
        dropright = DropFile(text='right')
        box.add_widget(dropright)
        return box

    def handledrops(self, *args):
        if False:
            for i in range(10):
                print('nop')
        for func in self.drops:
            func(*args)
DropApp().run()