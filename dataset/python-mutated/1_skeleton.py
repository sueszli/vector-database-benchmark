from kivy.app import App
from kivy.uix.widget import Widget

class MyPaintWidget(Widget):
    pass

class MyPaintApp(App):

    def build(self):
        if False:
            return 10
        return MyPaintWidget()
if __name__ == '__main__':
    MyPaintApp().run()