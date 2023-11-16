from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Rectangle, RoundedRectangle
from kivy.lang import Builder
TEXTURE = 'kiwi.jpg'
YELLOW = (1, 0.7, 0)
ORANGE = (1, 0.45, 0)
RED = (1, 0, 0)
WHITE = (1, 1, 1)

class RoundedRectangleWidget(Widget):

    def prepare(self):
        if False:
            while True:
                i = 10
        with self.canvas:
            Color(*WHITE)
            Rectangle(pos=(50, 400))
            RoundedRectangle(pos=(175, 400), radius=[0, 50, 0, 50], source=TEXTURE)
            Color(*YELLOW)
            RoundedRectangle(pos=(300, 400), radius=[0, 50, 0, 50])
            RoundedRectangle(pos=(425, 400), radius=[0, 50, 0, 50], source=TEXTURE)
            Color(*ORANGE)
            RoundedRectangle(pos=(50, 275), radius=[20])
            RoundedRectangle(pos=(175, 275), radius=[(20, 40)])
            Color(*RED)
            RoundedRectangle(pos=(300, 275), radius=[10, 20, 30, 40])
            RoundedRectangle(pos=(425, 275), radius=[(10, 20), (20, 30), (30, 40), (40, 50)])
            Color(*WHITE)
            Ellipse(pos=(50, 150))
            Ellipse(pos=(175, 150))
            Ellipse(pos=(300, 150))
            Ellipse(pos=(425, 150))
            RoundedRectangle(pos=(175, 150), radius=[9000], source=TEXTURE)
            Color(*RED)
            RoundedRectangle(pos=(300, 150), radius=[9000])
            RoundedRectangle(pos=(425, 150), radius=[9000], segments=15)
            Color(*ORANGE)
            RoundedRectangle(pos=(425, 150), radius=[9000], segments=2)
            Color(*YELLOW)
            RoundedRectangle(pos=(425, 150), radius=[9000], segments=1)
            RoundedRectangle(pos=(50, 25), radius=[40], segments=[1, 1, 10, 10], size=(125, 100))
            Color(*ORANGE)
            RoundedRectangle(pos=(200, 25), radius=[(40, 20), 45.5, 45.5, 0], segments=[2, 3, 3, 1], size=(125, 100))
            Color(*RED)
            RoundedRectangle(pos=(350, 25), radius=[(40, 40), (40, 40), (20, 20), (20, 20)], segments=[2, 3, 3, 2], size=(150, 100))

class DrawRoundedRectanglesApp(App):

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        kv = "\nWidget:\n    canvas:\n        Color:\n            rgba: 1, 1,1, 1\n\n        RoundedRectangle:\n            pos: 575, 400\n            size: 100, 100\n            radius: [0, 50, 0, 50]\n            source: 'kiwi.jpg'\n\n        Color:\n            rgba: 0, 0.8, 0.8, 1\n\n        RoundedRectangle:\n            pos: 575, 275\n            size: 100, 100\n            radius: [(10, 20), (20, 30), (30, 40), (40, 50)]\n\n        RoundedRectangle:\n            pos: 575, 150\n            size: 100, 100\n            radius: [9000]\n            segments: 15\n\n        RoundedRectangle:\n            pos: 550, 25\n            size: 150, 100\n            segments: [1, 2, 1, 3]\n            radius: [30, 40, 30, 40]\n\n"
        widget = RoundedRectangleWidget()
        widget.prepare()
        kvrect = Builder.load_string(kv)
        widget.add_widget(kvrect)
        return widget
if __name__ == '__main__':
    DrawRoundedRectanglesApp().run()