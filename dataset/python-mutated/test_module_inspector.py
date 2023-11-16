import unittest
from kivy.tests.common import GraphicUnitTest, UnitTestTouch
from kivy.base import EventLoop
from kivy.modules import inspector
from kivy.factory import Factory
KV = "\n#:import Factory kivy.factory.Factory\n\n<InspectorTestModal@ModalView>:\n    size_hint: 0.5, 0.5\n\n<FirstModal@InspectorTestModal>:\n    Button:\n        id: firstmodal\n        text: 'first modal'\n        on_release: Factory.SecondModal().open()\n\n<SecondModal@InspectorTestModal>:\n    Button:\n        id: secondmodal\n        text: 'second modal'\n        on_release: Factory.ThirdModal().open()\n\n<ThirdModal@InspectorTestModal>:\n    Button:\n        id: thirdmodal\n        text: 'third modal'\n\nBoxLayout:\n    Button:\n        id: highlight\n        text: 'highlight'\n    Button:\n        id: popup\n        text: 'popup'\n        on_release: Factory.FirstModal().open()\n    Button:\n        id: tri_popup\n        text: '3 popups'\n        on_release: Factory.FirstModal().open()\n    Button:\n        id: dummy\n        text: 'dummy'\n"

class InspectorTestCase(GraphicUnitTest):
    framecount = 0

    def setUp(self):
        if False:
            i = 10
            return i + 15
        import kivy.lang.builder as builder
        if not hasattr(self, '_trace'):
            self._trace = builder.trace
        self.builder = builder
        builder.trace = lambda *_, **__: None
        super(InspectorTestCase, self).setUp()

    def tearDown(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        import kivy.lang.builder as builder
        builder.Builder.unload_file('InspectorTestCase.KV')
        builder.trace = self._trace
        super(InspectorTestCase, self).tearDown(*args, **kwargs)

    def clean_garbage(self, *args):
        if False:
            while True:
                i = 10
        for child in self._win.children[:]:
            self._win.remove_widget(child)
        self.advance_frames(5)

    def test_activate_deactivate_bottom(self, *args):
        if False:
            print('Hello World!')
        EventLoop.ensure_window()
        self._win = EventLoop.window
        self.clean_garbage()
        self.root = self.builder.Builder.load_string(KV, filename='InspectorTestCase.KV')
        self.render(self.root)
        self.assertLess(len(self._win.children), 2)
        inspector.start(self._win, self.root)
        self.advance_frames(2)
        ins = self.root.inspector
        ins.activated = True
        ins.inspect_enabled = True
        self.assertTrue(ins.at_bottom)
        self.assertEqual(self._win.children[0], ins)
        self.advance_frames(1)
        self.assertLess(ins.layout.pos[1], self._win.height / 2.0)
        ins.inspect_enabled = False
        ins.activated = False
        self.render(self.root)
        self.advance_frames(1)
        inspector.stop(self._win, self.root)
        self.assertLess(len(self._win.children), 2)
        self.render(self.root)

    def test_activate_deactivate_top(self, *args):
        if False:
            i = 10
            return i + 15
        EventLoop.ensure_window()
        self._win = EventLoop.window
        self.clean_garbage()
        self.root = self.builder.Builder.load_string(KV, filename='InspectorTestCase.KV')
        self.render(self.root)
        self.assertLess(len(self._win.children), 2)
        inspector.start(self._win, self.root)
        self.advance_frames(2)
        ins = self.root.inspector
        ins.at_bottom = False
        ins.activated = True
        ins.inspect_enabled = True
        self.assertFalse(ins.at_bottom)
        self.assertEqual(self._win.children[0], ins)
        ins.toggle_position(self.root.ids.dummy)
        self.advance_frames(20)
        self.assertGreater(ins.layout.pos[1], self._win.height / 2.0)
        ins.inspect_enabled = False
        ins.activated = False
        self.render(self.root)
        self.advance_frames(1)
        inspector.stop(self._win, self.root)
        self.assertLess(len(self._win.children), 2)
        self.render(self.root)

    def test_widget_button(self, *args):
        if False:
            while True:
                i = 10
        EventLoop.ensure_window()
        self._win = EventLoop.window
        self.clean_garbage()
        self.root = self.builder.Builder.load_string(KV, filename='InspectorTestCase.KV')
        self.render(self.root)
        self.assertLess(len(self._win.children), 2)
        highlight = self.root.ids.highlight
        highlight_exp = self.root.ids.highlight.text
        inspector.start(self._win, self.root)
        self.advance_frames(2)
        ins = self.root.inspector
        ins.activated = True
        ins.inspect_enabled = True
        self.assertTrue(ins.at_bottom)
        touch = UnitTestTouch(*highlight.center)
        touch.touch_down()
        touch.touch_up()
        ins.show_widget_info()
        self.advance_frames(2)
        self.assertEqual(ins.widget.text, highlight_exp)
        for node in ins.treeview.iterate_all_nodes():
            lkey = getattr(node.ids, 'lkey', None)
            if not lkey:
                continue
            if lkey.text == 'text':
                ltext = node.ids.ltext
                self.assertEqual(ltext.text[1:-1], highlight_exp)
                break
        ins.inspect_enabled = False
        ins.activated = False
        self.render(self.root)
        self.advance_frames(1)
        inspector.stop(self._win, self.root)
        self.assertLess(len(self._win.children), 2)
        self.render(self.root)

    def test_widget_popup(self, *args):
        if False:
            i = 10
            return i + 15
        EventLoop.ensure_window()
        self._win = EventLoop.window
        self.clean_garbage()
        self.root = self.builder.Builder.load_string(KV, filename='InspectorTestCase.KV')
        self.render(self.root)
        self.assertLess(len(self._win.children), 2)
        popup = self.root.ids.popup
        inspector.start(self._win, self.root)
        self.advance_frames(1)
        ins = self.root.inspector
        ins.inspect_enabled = False
        ins.activated = True
        self.assertTrue(ins.at_bottom)
        touch = UnitTestTouch(*popup.center)
        touch.touch_down()
        touch.touch_up()
        self.advance_frames(1)
        ins.inspect_enabled = True
        self.advance_frames(1)
        touch.touch_down()
        touch.touch_up()
        self.advance_frames(1)
        ins.show_widget_info()
        self.advance_frames(2)
        self.assertIsInstance(ins.widget, Factory.Button)
        self.assertIsInstance(ins.widget.parent, Factory.FirstModal)
        temp_popup = Factory.FirstModal()
        temp_popup_exp = temp_popup.ids.firstmodal.text
        self.assertEqual(ins.widget.text, temp_popup_exp)
        for node in ins.treeview.iterate_all_nodes():
            lkey = getattr(node.ids, 'lkey', None)
            if not lkey:
                continue
            if lkey.text == 'text':
                ltext = node.ids.ltext
                self.assertEqual(ltext.text[1:-1], temp_popup_exp)
                break
        del temp_popup
        ins.inspect_enabled = False
        touch = UnitTestTouch(0, 0)
        touch.touch_down()
        touch.touch_up()
        self.advance_frames(10)
        ins.activated = False
        self.render(self.root)
        self.advance_frames(5)
        inspector.stop(self._win, self.root)
        self.assertLess(len(self._win.children), 2)
        self.render(self.root)

    @unittest.skip("doesn't work on CI with Python 3.5 but works locally")
    def test_widget_multipopup(self, *args):
        if False:
            i = 10
            return i + 15
        EventLoop.ensure_window()
        self._win = EventLoop.window
        self.clean_garbage()
        self.root = self.builder.Builder.load_string(KV, filename='InspectorTestCase.KV')
        self.render(self.root)
        self.assertLess(len(self._win.children), 2)
        popup = self.root.ids.popup
        inspector.start(self._win, self.root)
        self.advance_frames(1)
        ins = self.root.inspector
        ins.inspect_enabled = False
        ins.activated = True
        self.assertTrue(ins.at_bottom)
        touch = UnitTestTouch(*popup.center)
        touch.touch_down()
        touch.touch_up()
        self.advance_frames(1)
        touch = UnitTestTouch(self._win.width / 2.0, self._win.height / 2.0)
        for i in range(2):
            touch.touch_down()
            touch.touch_up()
            self.advance_frames(1)
        modals = [Factory.ThirdModal, Factory.SecondModal, Factory.FirstModal]
        for mod in modals:
            ins.inspect_enabled = True
            self.advance_frames(1)
            touch.touch_down()
            touch.touch_up()
            self.advance_frames(1)
            self.assertIsInstance(ins.widget, Factory.Button)
            self.assertIsInstance(ins.widget.parent, mod)
            ins.inspect_enabled = False
            orig = UnitTestTouch(0, 0)
            orig.touch_down()
            orig.touch_up()
            self.advance_frames(10)
        ins.activated = False
        self.render(self.root)
        self.advance_frames(5)
        inspector.stop(self._win, self.root)
        self.assertLess(len(self._win.children), 2)
        self.render(self.root)
if __name__ == '__main__':
    import unittest
    unittest.main()