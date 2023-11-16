"""
Box layout unit test
====================

Order matter.
On the screen, most of example must have the red->blue->green order.
"""
from kivy.tests.common import GraphicUnitTest

class UIXBoxLayoutTestcase(GraphicUnitTest):

    def box(self, r, g, b):
        if False:
            return 10
        from kivy.uix.widget import Widget
        from kivy.graphics import Color, Rectangle
        wid = Widget()
        with wid.canvas:
            Color(r, g, b)
            r = Rectangle(pos=wid.pos, size=wid.size)

        def linksp(instance, *largs):
            if False:
                print('Hello World!')
            r.pos = instance.pos
            r.size = instance.size
        wid.bind(pos=linksp, size=linksp)
        return wid

    def test_boxlayout_orientation(self):
        if False:
            for i in range(10):
                print('nop')
        from kivy.uix.boxlayout import BoxLayout
        r = self.render
        b = self.box
        layout = BoxLayout()
        layout.add_widget(b(1, 0, 0))
        layout.add_widget(b(0, 1, 0))
        layout.add_widget(b(0, 0, 1))
        r(layout)
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(b(1, 0, 0))
        layout.add_widget(b(0, 1, 0))
        layout.add_widget(b(0, 0, 1))
        r(layout)

    def test_boxlayout_spacing(self):
        if False:
            print('Hello World!')
        from kivy.uix.boxlayout import BoxLayout
        r = self.render
        b = self.box
        layout = BoxLayout(spacing=20)
        layout.add_widget(b(1, 0, 0))
        layout.add_widget(b(0, 1, 0))
        layout.add_widget(b(0, 0, 1))
        r(layout)
        layout = BoxLayout(spacing=20, orientation='vertical')
        layout.add_widget(b(1, 0, 0))
        layout.add_widget(b(0, 1, 0))
        layout.add_widget(b(0, 0, 1))
        r(layout)

    def test_boxlayout_padding(self):
        if False:
            while True:
                i = 10
        from kivy.uix.boxlayout import BoxLayout
        r = self.render
        b = self.box
        layout = BoxLayout(padding=20)
        layout.add_widget(b(1, 0, 0))
        layout.add_widget(b(0, 1, 0))
        layout.add_widget(b(0, 0, 1))
        r(layout)
        layout = BoxLayout(padding=20, orientation='vertical')
        layout.add_widget(b(1, 0, 0))
        layout.add_widget(b(0, 1, 0))
        layout.add_widget(b(0, 0, 1))
        r(layout)

    def test_boxlayout_padding_spacing(self):
        if False:
            i = 10
            return i + 15
        from kivy.uix.boxlayout import BoxLayout
        r = self.render
        b = self.box
        layout = BoxLayout(spacing=20, padding=20)
        layout.add_widget(b(1, 0, 0))
        layout.add_widget(b(0, 1, 0))
        layout.add_widget(b(0, 0, 1))
        r(layout)
        layout = BoxLayout(spacing=20, padding=20, orientation='vertical')
        layout.add_widget(b(1, 0, 0))
        layout.add_widget(b(0, 1, 0))
        layout.add_widget(b(0, 0, 1))
        r(layout)