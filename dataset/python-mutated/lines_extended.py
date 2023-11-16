"""
Lines Extended Demo
===================

This demonstrates how to use the extended line drawing routines such
as circles, ellipses, and rectangles. You should see a static image of
labelled shapes on the screen.
"""
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.lang import Builder
Builder.load_string("\n<LineEllipse1>:\n    canvas:\n        Color:\n            rgba: 1, .1, .1, .9\n        Line:\n            width: 2.\n            ellipse: (self.x, self.y, self.width, self.height)\n    Label:\n        center: root.center\n        text: 'Ellipse'\n\n<LineEllipse2>:\n    canvas:\n        Color:\n            rgba: 1, .1, .1, .9\n        Line:\n            width: 2.\n            ellipse: (self.x, self.y, self.width, self.height, 90, 180)\n    Label:\n        center: root.center\n        text: 'Ellipse from 90 to 180'\n\n# fun result with low segments!\n<LineEllipse3>:\n    canvas:\n        Color:\n            rgba: 1, .1, .1, .9\n        Line:\n            width: 2.\n            ellipse: (self.x, self.y, self.width, self.height, 90, 720, 10)\n    Label:\n        center: root.center\n        text: 'Ellipse from 90 to 720\\n10 segments'\n        halign: 'center'\n\n<LineCircle1>:\n    canvas:\n        Color:\n            rgba: .1, 1, .1, .9\n        Line:\n            width: 2.\n            circle:\n                (self.center_x, self.center_y, min(self.width, self.height)\n                / 2)\n    Label:\n        center: root.center\n        text: 'Circle'\n\n<LineCircle2>:\n    canvas:\n        Color:\n            rgba: .1, 1, .1, .9\n        Line:\n            width: 2.\n            circle:\n                (self.center_x, self.center_y, min(self.width, self.height)\n                / 2, 90, 180)\n    Label:\n        center: root.center\n        text: 'Circle from 90 to 180'\n\n<LineCircle3>:\n    canvas:\n        Color:\n            rgba: .1, 1, .1, .9\n        Line:\n            width: 2.\n            circle:\n                (self.center_x, self.center_y, min(self.width, self.height)\n                / 2, 90, 180, 10)\n    Label:\n        center: root.center\n        text: 'Circle from 90 to 180\\n10 segments'\n        halign: 'center'\n\n<LineCircle4>:\n    canvas:\n        Color:\n            rgba: .1, 1, .1, .9\n        Line:\n            width: 2.\n            circle:\n                (self.center_x, self.center_y, min(self.width, self.height)\n                / 2, 0, 360)\n    Label:\n        center: root.center\n        text: 'Circle from 0 to 360'\n        halign: 'center'\n\n<LineRectangle>:\n    canvas:\n        Color:\n            rgba: .1, .1, 1, .9\n        Line:\n            width: 2.\n            rectangle: (self.x, self.y, self.width, self.height)\n    Label:\n        center: root.center\n        text: 'Rectangle'\n\n<LineBezier>:\n    canvas:\n        Color:\n            rgba: .1, .1, 1, .9\n        Line:\n            width: 2.\n            bezier:\n                (self.x, self.y, self.center_x - 40, self.y + 100,\n                self.center_x + 40, self.y - 100, self.right, self.y)\n    Label:\n        center: root.center\n        text: 'Bezier'\n")

class LineEllipse1(Widget):
    pass

class LineEllipse2(Widget):
    pass

class LineEllipse3(Widget):
    pass

class LineCircle1(Widget):
    pass

class LineCircle2(Widget):
    pass

class LineCircle3(Widget):
    pass

class LineCircle4(Widget):
    pass

class LineRectangle(Widget):
    pass

class LineBezier(Widget):
    pass

class LineExtendedApp(App):

    def build(self):
        if False:
            print('Hello World!')
        root = GridLayout(cols=2, padding=50, spacing=50)
        root.add_widget(LineEllipse1())
        root.add_widget(LineEllipse2())
        root.add_widget(LineEllipse3())
        root.add_widget(LineCircle1())
        root.add_widget(LineCircle2())
        root.add_widget(LineCircle3())
        root.add_widget(LineCircle4())
        root.add_widget(LineRectangle())
        root.add_widget(LineBezier())
        return root
if __name__ == '__main__':
    LineExtendedApp().run()