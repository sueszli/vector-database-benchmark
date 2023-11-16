from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        if False:
            for i in range(10):
                print('nop')
        with self.canvas:
            Color(1, 1, 0)
            d = 30.0
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))

class MyPaintApp(App):

    def build(self):
        if False:
            print('Hello World!')
        return MyPaintWidget()
if __name__ == '__main__':
    MyPaintApp().run()