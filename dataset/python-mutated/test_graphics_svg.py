"""
Svg tests
==============

Testing Svg rendering.
"""
from kivy.tests.common import GraphicUnitTest
SIMPLE_SVG = '<?xml version="1.0" standalone="no"?>\n<svg width="256" height="256" viewBox="0 0 256 256" version="1.1"\n    xmlns="http://www.w3.org/2000/svg">\n<rect stroke="blue" stroke-width="4" x="24" y="30" width="92" height="166"\n    fill="none" stroke-opacity="0.5" />\n</svg>\n'
SCALE_SVG = '<?xml version="1.0" standalone="no"?>\n<svg width="256" height="256" viewBox="0 0 256 256" version="1.1"\n    xmlns="http://www.w3.org/2000/svg">\n<rect stroke="red" stroke-width="4" x="24" y="30" width="10" height="10"\n    fill="none" stroke-opacity="0.5" transform="scale(2, 3)"/>\n</svg>\n'
ROTATE_SVG = '<?xml version="1.0" standalone="no"?>\n<svg width="256" height="256" viewBox="0 0 256 256" version="1.1"\n    xmlns="http://www.w3.org/2000/svg">\n<rect stroke="green" stroke-width="4" x="24" y="30" width="50" height="100"\n    stroke-opacity="0.75" transform="rotate(60 128 128)" />\n</svg>\n'

class SvgTest(GraphicUnitTest):

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        import xml.etree.ElementTree as ET
        from kivy.uix.widget import Widget
        from kivy.graphics.svg import Svg
        wid = Widget()
        with wid.canvas:
            svg = Svg()
            svg.set_tree(ET.ElementTree(ET.fromstring(SIMPLE_SVG)))
        self.render(wid)

    def test_scale(self):
        if False:
            print('Hello World!')
        import xml.etree.ElementTree as ET
        from kivy.uix.widget import Widget
        from kivy.graphics.svg import Svg
        wid = Widget()
        with wid.canvas:
            svg = Svg()
            svg.set_tree(ET.ElementTree(ET.fromstring(SCALE_SVG)))
        self.render(wid)

    def test_rotate(self):
        if False:
            for i in range(10):
                print('nop')
        import xml.etree.ElementTree as ET
        from kivy.uix.widget import Widget
        from kivy.graphics.svg import Svg
        wid = Widget()
        with wid.canvas:
            svg = Svg()
            svg.set_tree(ET.ElementTree(ET.fromstring(ROTATE_SVG)))
        self.render(wid)