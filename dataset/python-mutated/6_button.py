from random import random
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        if False:
            print('Hello World!')
        color = (random(), 1, 1)
        with self.canvas:
            Color(*color, mode='hsv')
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
            print('Hello World!')
        parent = Widget()
        self.painter = MyPaintWidget()
        clearbtn = Button(text='Clear')
        clearbtn.bind(on_release=self.clear_canvas)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        return parent

    def clear_canvas(self, obj):
        if False:
            for i in range(10):
                print('nop')
        self.painter.canvas.clear()
if __name__ == '__main__':
    MyPaintApp().run()