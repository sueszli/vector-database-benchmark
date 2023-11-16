from kivy.app import App
from kivy.uix.widget import Widget

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        if False:
            i = 10
            return i + 15
        print(touch)

class MyPaintApp(App):

    def build(self):
        if False:
            return 10
        return MyPaintWidget()
if __name__ == '__main__':
    MyPaintApp().run()