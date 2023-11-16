"""Tests for Linux AtspiWrapper"""
import os
import sys
import unittest
if sys.platform.startswith('linux'):
    sys.path.append('.')
    from pywinauto.linux.atspi_element_info import AtspiElementInfo
    from pywinauto.linux.application import Application
    from pywinauto.controls.atspiwrapper import AtspiWrapper
    from pywinauto.linux.atspi_objects import IATSPI
    from pywinauto.linux.atspi_objects import POINT
app_name = 'gtk_example.py'

def _test_app():
    if False:
        for i in range(10):
            print('nop')
    test_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'apps/Gtk_samples')
    sys.path.append(test_folder)
    return os.path.join(test_folder, app_name)

def print_tree(start_el_info, level_shifter=''):
    if False:
        i = 10
        return i + 15
    if level_shifter == '':
        print(start_el_info.control_type, '  ', start_el_info.control_id, '!')
        level_shifter += '-'
    for children in start_el_info.children():
        print(level_shifter, '  ', children.control_type, '    ', children.control_id, children.runtime_id, '!')
        print_tree(children, level_shifter + '-')
if sys.platform.startswith('linux'):

    class AtspiWrapperTests(unittest.TestCase):
        """Unit tests for the AtspiWrapper class"""

        def setUp(self):
            if False:
                return 10
            self.desktop_info = AtspiElementInfo()
            self.desktop_wrapper = AtspiWrapper(self.desktop_info)
            self.app = Application()
            self.app.start(_test_app())
            self.app_wrapper = self.app.gtk_example.find()
            self.app_frame = self.app.gtk_example.Frame

        def tearDown(self):
            if False:
                for i in range(10):
                    print('nop')
            self.app.kill()

        def test_set_focus(self):
            if False:
                i = 10
                return i + 15
            states = self.app_frame.set_focus().get_states()
            self.assertIn('STATE_VISIBLE', states)
            self.assertIn('STATE_SHOWING', states)

        def test_top_level_parent_for_app_return_app(self):
            if False:
                return 10
            self.assertEqual(self.app_wrapper.top_level_parent().element_info.control_type, 'Application')

        def test_top_level_parent_for_button_return_app(self):
            if False:
                return 10
            self.assertEqual(self.app_frame.Panel.top_level_parent().element_info.control_type, 'Application')

        def test_root_return_desktop(self):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(self.app_wrapper.root(), self.desktop_info)

        def test_class_name_return_element_info_class_name(self):
            if False:
                print('Hello World!')
            self.assertEqual(self.app_wrapper.class_name(), 'Application')

        def test_window_text(self):
            if False:
                print('Hello World!')
            self.assertEqual(self.app_wrapper.window_text(), app_name)

        def test_control_id(self):
            if False:
                return 10
            self.assertEqual(self.app_wrapper.control_id(), IATSPI().known_control_types['Application'])

        def test_image(self):
            if False:
                for i in range(10):
                    print('nop')
            img_wrp = self.app_frame.Icon.find()
            self.assertEqual(img_wrp.description(), u'')
            self.assertEqual(img_wrp.class_name(), u'Icon')
            self.assertEqual(img_wrp.locale(), u'')
            self.assertEqual(img_wrp.size(), (48, 24))
            pos = img_wrp.position()
            self.assertAlmostEqual(pos.x, 408, delta=5)
            self.assertAlmostEqual(pos.y, 29, delta=5)
            bb = img_wrp.bounding_box()
            self.assertEqual(bb.left, pos.x)
            self.assertEqual(bb.top, pos.y)
            self.assertAlmostEqual(bb.right, 456, delta=5)
            self.assertAlmostEqual(bb.bottom, 53, delta=5)

        def test_can_get_rectangle(self):
            if False:
                print('Hello World!')
            rect = self.app_frame.Panel.rectangle()
            self.assertEqual(rect.width(), 600)
            rect = self.app_frame.Icon.rectangle()
            self.assertAlmostEqual(rect.height(), 26, delta=2)

        def test_client_to_screen(self):
            if False:
                for i in range(10):
                    print('nop')
            rect = self.app_wrapper.rectangle()
            self.assertEqual(self.app_wrapper.client_to_screen((0, 0)), (rect.left, rect.top))
            self.assertEqual(self.app_wrapper.client_to_screen(POINT(20, 20)), (rect.left + 20, rect.top + 20))

        def test_can_get_process_id(self):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(self.app_wrapper.process_id(), self.app.process)

        def test_is_dialog_for_Application_is_true(self):
            if False:
                while True:
                    i = 10
            self.assertTrue(self.app_wrapper.is_dialog())

        def test_is_dialog_for_button_is_false(self):
            if False:
                i = 10
                return i + 15
            self.assertFalse(self.app_frame.Panel.Click.is_dialog())

        def test_can_get_children(self):
            if False:
                i = 10
                return i + 15
            self.assertEqual(self.app_frame.control_id(), IATSPI().known_control_types['Frame'])

        def test_can_get_descendants(self):
            if False:
                while True:
                    i = 10
            self.assertTrue(len(self.app_wrapper.descendants()) > len(self.app_wrapper.children()))

        def test_can_get_control_count(self):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(self.app_wrapper.control_count(), 1)

        def test_can_get_properties(self):
            if False:
                i = 10
                return i + 15
            props = self.app_wrapper.get_properties()
            self.assertEqual(props['class_name'], 'Application')
            self.assertEqual(props['friendly_class_name'], 'Application')
            self.assertEqual(props['control_id'], IATSPI().known_control_types['Application'])

        def test_app_is_child_of_desktop(self):
            if False:
                print('Hello World!')
            self.assertTrue(self.app_wrapper.is_child(self.desktop_wrapper))
if __name__ == '__main__':
    unittest.main()