from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Ellipse, Line
from kivy.gesture import Gesture, GestureDatabase
from my_gestures import cross, circle, check, square

def simplegesture(name, point_list):
    if False:
        i = 10
        return i + 15
    '\n    A simple helper function\n    '
    g = Gesture()
    g.add_stroke(point_list)
    g.normalize()
    g.name = name
    return g

class GestureBoard(FloatLayout):
    """
    Our application main widget, derived from touchtracer example, use data
    constructed from touches to match symbols loaded from my_gestures.

    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(GestureBoard, self).__init__()
        self.gdb = GestureDatabase()
        self.gdb.add_gesture(cross)
        self.gdb.add_gesture(check)
        self.gdb.add_gesture(circle)
        self.gdb.add_gesture(square)

    def on_touch_down(self, touch):
        if False:
            i = 10
            return i + 15
        userdata = touch.ud
        with self.canvas:
            Color(1, 1, 0)
            d = 30.0
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            userdata['line'] = Line(points=(touch.x, touch.y))
        return True

    def on_touch_move(self, touch):
        if False:
            for i in range(10):
                print('nop')
        try:
            touch.ud['line'].points += [touch.x, touch.y]
            return True
        except KeyError as e:
            pass

    def on_touch_up(self, touch):
        if False:
            i = 10
            return i + 15
        g = simplegesture('', list(zip(touch.ud['line'].points[::2], touch.ud['line'].points[1::2])))
        print('gesture representation:', self.gdb.gesture_to_str(g))
        print('cross:', g.get_score(cross))
        print('check:', g.get_score(check))
        print('circle:', g.get_score(circle))
        print('square:', g.get_score(square))
        g2 = self.gdb.find(g, minscore=0.7)
        print(g2)
        if g2:
            if g2[1] == circle:
                print('circle')
            if g2[1] == square:
                print('square')
            if g2[1] == check:
                print('check')
            if g2[1] == cross:
                print('cross')
        self.canvas.clear()

class DemoGesture(App):

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        return GestureBoard()
if __name__ == '__main__':
    DemoGesture().run()