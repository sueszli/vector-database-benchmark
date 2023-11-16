from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        if False:
            print('Hello World!')
        with self.canvas:
            Color(1, 1, 0)
            d = 30.0
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            touch.ud['line'] = Line(points=(touch.x, touch.y))

    def on_touch_move(self, touch):
        if False:
            return 10
        touch.ud['line'].points += [touch.x, touch.y]

class MyPaintApp(App):

    def build(self):
        if False:
            return 10
        return MyPaintWidget()
if __name__ == '__main__':
    MyPaintApp().run()