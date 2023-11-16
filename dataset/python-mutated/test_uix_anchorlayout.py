"""
Anchor layout unit test
=======================
"""
from kivy.tests.common import GraphicUnitTest

class UIXAnchorLayoutTestcase(GraphicUnitTest):

    def box(self, r, g, b):
        if False:
            print('Hello World!')
        from kivy.uix.widget import Widget
        from kivy.graphics import Color, Rectangle
        wid = Widget(size_hint=(None, None), size=(100, 100))
        with wid.canvas:
            Color(r, g, b)
            r = Rectangle(pos=wid.pos, size=wid.size)

        def linksp(instance, *largs):
            if False:
                for i in range(10):
                    print('nop')
            r.pos = instance.pos
            r.size = instance.size
        wid.bind(pos=linksp, size=linksp)
        return wid

    def test_anchorlayout_default(self):
        if False:
            for i in range(10):
                print('nop')
        from kivy.uix.anchorlayout import AnchorLayout
        r = self.render
        b = self.box
        layout = AnchorLayout()
        layout.add_widget(b(1, 0, 0))
        r(layout)

    def test_anchorlayout_x(self):
        if False:
            i = 10
            return i + 15
        from kivy.uix.anchorlayout import AnchorLayout
        r = self.render
        b = self.box
        layout = AnchorLayout(anchor_x='left')
        layout.add_widget(b(1, 0, 0))
        r(layout)
        layout = AnchorLayout(anchor_x='center')
        layout.add_widget(b(1, 0, 0))
        r(layout)
        layout = AnchorLayout(anchor_x='right')
        layout.add_widget(b(1, 0, 0))
        r(layout)

    def test_anchorlayout_y(self):
        if False:
            for i in range(10):
                print('nop')
        from kivy.uix.anchorlayout import AnchorLayout
        r = self.render
        b = self.box
        layout = AnchorLayout(anchor_y='bottom')
        layout.add_widget(b(1, 0, 0))
        r(layout)
        layout = AnchorLayout(anchor_y='center')
        layout.add_widget(b(1, 0, 0))
        r(layout)
        layout = AnchorLayout(anchor_y='top')
        layout.add_widget(b(1, 0, 0))
        r(layout)

    def test_anchor_layout_xy(self):
        if False:
            i = 10
            return i + 15
        from kivy.uix.anchorlayout import AnchorLayout
        r = self.render
        b = self.box
        layout = AnchorLayout(anchor_y='bottom', anchor_x='left')
        layout.add_widget(b(1, 0, 0))
        r(layout)
        layout = AnchorLayout(anchor_y='top', anchor_x='right')
        layout.add_widget(b(1, 0, 0))
        r(layout)