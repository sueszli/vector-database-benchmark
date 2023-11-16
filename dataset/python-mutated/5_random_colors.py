from random import random
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        if False:
            return 10
        color = (random(), random(), random())
        with self.canvas:
            Color(*color)
            d = 30.0
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            touch.ud['line'] = Line(points=(touch.x, touch.y))

    def on_touch_move(self, touch):
        if False:
            for i in range(10):
                print('nop')
        touch.ud['line'].points += [touch.x, touch.y]

class MyPaintApp(App):

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        return MyPaintWidget()
if __name__ == '__main__':
    MyPaintApp().run()