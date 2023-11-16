"""Tests for Linux AtspiElementInfo"""
import os
import sys
import subprocess
import time
import unittest
import re
import mock
if sys.platform.startswith('linux'):
    sys.path.append('.')
    from pywinauto.linux.atspi_objects import AtspiAccessible
    from pywinauto.linux.atspi_objects import _AtspiRect
    from pywinauto.linux.atspi_objects import RECT
    from pywinauto.linux.atspi_objects import _AtspiPoint
    from pywinauto.linux.atspi_objects import POINT
    from pywinauto.linux.atspi_objects import _AtspiCoordType
    from pywinauto.linux.atspi_element_info import AtspiElementInfo
    from pywinauto.linux.atspi_objects import IATSPI
    from pywinauto.linux.application import Application
    from pywinauto.linux.atspi_objects import GHashTable
    from pywinauto.linux.atspi_objects import _find_library
app_name = 'gtk_example.py'

def _test_app():
    if False:
        i = 10
        return i + 15
    test_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'apps/Gtk_samples')
    sys.path.append(test_folder)
    return os.path.join(test_folder, app_name)
if sys.platform.startswith('linux'):

    class AtspiPointTests(unittest.TestCase):
        """Unit tests for AtspiPoint class"""

        def test_indexation(self):
            if False:
                print('Hello World!')
            p = POINT(1, 2)
            self.assertEqual(p[0], p.x)
            self.assertEqual(p[1], p.y)
            self.assertEqual(p[-2], p.x)
            self.assertEqual(p[-1], p.y)
            self.assertRaises(IndexError, lambda : p[2])
            self.assertRaises(IndexError, lambda : p[-3])

        def test_iteration(self):
            if False:
                for i in range(10):
                    print('nop')
            p = POINT(1, 2)
            self.assertEqual([1, 2], [i for i in p])

        def test_equal(self):
            if False:
                i = 10
                return i + 15
            p0 = POINT(1, 2)
            p1 = POINT(0, 2)
            self.assertNotEqual(p0, p1)
            p1.x = p0.x
            self.assertEqual(p0, p1)

    class AtspiRectTests(unittest.TestCase):
        """Unit tests for AtspiRect class"""

        def test_RECT_hash(self):
            if False:
                print('Hello World!')
            'Test RECT is not hashable'
            self.assertRaises(TypeError, hash, RECT())

        def test_RECT_eq(self):
            if False:
                print('Hello World!')
            r0 = RECT(1, 2, 3, 4)
            self.assertEqual(r0, RECT(1, 2, 3, 4))
            self.assertEqual(r0, [1, 2, 3, 4])
            self.assertNotEqual(r0, RECT(1, 2, 3, 5))
            self.assertNotEqual(r0, [1, 2, 3, 5])
            self.assertNotEqual(r0, [1, 2, 3])
            self.assertNotEqual(r0, [1, 2, 3, 4, 5])
            r0.bottom = 5
            self.assertEqual(r0, RECT(1, 2, 3, 5))
            self.assertEqual(r0, (1, 2, 3, 5))

        def test_RECT_repr(self):
            if False:
                for i in range(10):
                    print('nop')
            'Test RECT repr'
            r0 = RECT(0)
            self.assertEqual(r0.__repr__(), '<RECT L0, T0, R0, B0>')

        def test_RECT_iter(self):
            if False:
                print('Hello World!')
            'Test RECT is iterable'
            r = RECT(1, 2, 3, 4)
            (left, top, right, bottom) = r
            self.assertEqual(left, r.left)
            self.assertEqual(right, r.right)
            self.assertEqual(top, r.top)
            self.assertEqual(bottom, r.bottom)

    class AtspiElementInfoTests(unittest.TestCase):
        """Unit tests for the AtspiElementInfo class"""

        def get_app(self, name, pid=None):
            if False:
                while True:
                    i = 10
            'Helper to find the application top window'
            if not pid:
                pid = self.app.process
            for child in self.desktop_info.children():
                if child.name == name and pid == child.process_id:
                    return child
            raise Exception('Application not found')

        def setUp(self):
            if False:
                return 10
            self.desktop_info = AtspiElementInfo()
            self.app = Application()
            self.app.start(_test_app())
            time.sleep(1)
            self.app_info = self.get_app(app_name)
            self.app2 = None

        def tearDown(self):
            if False:
                return 10
            self.app.kill()
            if self.app2:
                self.app2.kill()

        def test_can_get_desktop(self):
            if False:
                i = 10
                return i + 15
            self.assertEqual(self.desktop_info.control_type, 'DesktopFrame')

        def test_can_get_childrens(self):
            if False:
                while True:
                    i = 10
            apps = [children.name for children in self.desktop_info.children()]
            self.assertTrue(app_name in apps)

        def test_can_get_name(self):
            if False:
                i = 10
                return i + 15
            self.assertEqual(self.desktop_info.name, 'main')

        def test_can_get_parent(self):
            if False:
                while True:
                    i = 10
            parent = self.app_info.parent
            self.assertEqual(parent.control_type, 'DesktopFrame')

        def test_can_get_process_id(self):
            if False:
                i = 10
                return i + 15
            self.assertEqual(self.app_info.process_id, self.app.process)

        def test_can_get_class_name(self):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(self.app_info.class_name, 'Application')

        def test_can_get_control_type_property(self):
            if False:
                i = 10
                return i + 15
            self.assertEqual(self.app_info.control_type, 'Application')

        def test_can_get_control_type_of_all_app_descendants(self):
            if False:
                return 10
            children = self.app_info.descendants()
            self.assertNotEqual(len(children), 0)
            print(children)
            for child in children:
                self.assertTrue(child.control_type in IATSPI().known_control_types.keys())

        def test_control_type_equal_class_name(self):
            if False:
                return 10
            children = self.app_info.descendants()
            self.assertNotEqual(len(children), 0)
            for child in children:
                self.assertEqual(child.control_type, child.class_name)

        def test_can_get_description(self):
            if False:
                return 10
            self.assertEqual(self.app_info.description(), '')

        def test_can_get_framework_id(self):
            if False:
                print('Hello World!')
            dpkg_output = subprocess.check_output(['dpkg', '-s', 'libgtk-3-0']).decode(encoding='UTF-8')
            version_line = None
            for line in dpkg_output.split('\n'):
                if line.startswith('Version'):
                    version_line = line
                    break
            print(version_line)
            if version_line is None:
                raise Exception('Cant get system gtk version')
            version_pattern = 'Version:\\s+(\\d+\\.\\d+\\.\\d+).*'
            r_version = re.compile(version_pattern, flags=re.MULTILINE)
            res = r_version.match(version_line)
            gtk_version = res.group(1)
            self.assertEqual(self.app_info.framework_id(), gtk_version)

        def test_can_get_framework_name(self):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(self.app_info.framework_name(), 'gtk')

        def test_can_get_atspi_version(self):
            if False:
                print('Hello World!')
            version = self.app_info.atspi_version()
            self.assertTrue(version in ['2.0', '2.1'], msg='Unexpected AT-SPI version: {}'.format(version))

        def test_can_get_rectangle(self):
            if False:
                while True:
                    i = 10
            app_info = self.get_app(app_name)
            frame = app_info.children()[0]
            filler = frame.children()[0]
            rectangle = filler.rectangle
            self.assertEqual(rectangle.width(), 600)
            self.assertAlmostEqual(rectangle.height(), 450, delta=50)

        def test_can_compare_applications(self):
            if False:
                print('Hello World!')
            app_info = self.get_app(app_name)
            app_info1 = self.get_app(app_name)
            self.assertEqual(app_info, app_info1)
            self.assertNotEqual(app_info, None)
            self.assertNotEqual(app_info, app_info.handle)
            self.assertNotEqual(app_info, AtspiElementInfo())

        def test_can_compare_desktop(self):
            if False:
                for i in range(10):
                    print('nop')
            desktop = AtspiElementInfo()
            desktop1 = AtspiElementInfo()
            self.assertEqual(desktop, desktop1)

        def test_can_get_layer(self):
            if False:
                print('Hello World!')
            self.assertEqual(self.desktop_info.get_layer(), 3)

        def test_can_get_order(self):
            if False:
                return 10
            self.assertEqual(self.desktop_info.get_order(), 0)

        def test_can_get_state_set(self):
            if False:
                print('Hello World!')
            frame_info = self.app_info.children()[0]
            states = frame_info.get_state_set()
            self.assertIn('STATE_VISIBLE', states)

        def test_visible(self):
            if False:
                return 10
            frame_info = self.app_info.children()[0]
            self.assertTrue(frame_info.visible)

        def test_enabled(self):
            if False:
                for i in range(10):
                    print('nop')
            frame_info = self.app_info.children()[0]
            self.assertTrue(frame_info.enabled)

        def test_hash(self):
            if False:
                print('Hello World!')
            self.app2 = Application().start(_test_app())
            time.sleep(1)
            app_info2 = self.get_app(app_name, pid=self.app2.process)
            frame_info1 = self.app_info.children()[0]
            frame_info2 = app_info2.children()[0]
            d = {frame_info1: 1, frame_info2: 2}
            self.assertEqual(d[frame_info1], d[self.app_info.children()[0]])
            self.assertNotEqual(d[frame_info1], d[frame_info2])
            self.assertEqual(d[frame_info2], d[frame_info2])

    class AtspiElementInfoWithoutChildrenMockedTests(unittest.TestCase):
        """Mocked unit tests for the AtspiElementInfo without children"""

        def setUp(self):
            if False:
                print('Hello World!')
            self.info = AtspiElementInfo()
            self.patch_get_child_count = mock.patch.object(AtspiAccessible, 'get_child_count')
            self.mock_get_child_count = self.patch_get_child_count.start()
            self.mock_get_child_count.return_value = 0
            self.patch_get_role = mock.patch.object(AtspiAccessible, 'get_role')
            self.mock_get_role = self.patch_get_role.start()
            self.mock_get_role.return_value = IATSPI().known_control_types['Application']

        def tearDown(self):
            if False:
                print('Hello World!')
            self.patch_get_role.stop()
            self.patch_get_child_count.stop()

        def test_service_is_not_visible(self):
            if False:
                return 10
            self.assertFalse(self.info.visible)

    class AtspiElementInfoMockedTests(unittest.TestCase):
        """Various mocked unit tests for the AtspiElementInfo"""

        def setUp(self):
            if False:
                i = 10
                return i + 15
            pass

        def tearDown(self):
            if False:
                while True:
                    i = 10
            pass

        def test_control_type_exception_on_bad_role_id(self):
            if False:
                for i in range(10):
                    print('nop')
            with mock.patch.object(AtspiAccessible, 'get_role') as mock_get_role:
                mock_get_role.return_value = 3735928559
                info = AtspiElementInfo()
                with self.assertRaises(NotImplementedError):
                    info.control_type()

    class GHashTableTests(unittest.TestCase):
        """Tests manipulating native GHashTable"""

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def tearDown(self):
            if False:
                while True:
                    i = 10
            pass

        def test_ghash2dic(self):
            if False:
                i = 10
                return i + 15
            'Test handling C-created ghash_table with string-based KV pairs'
            ghash_table_p = GHashTable.dic2ghash({b'key1': b'val1', b'2key': b'value2'})
            dic = GHashTable.ghash2dic(ghash_table_p)
            print(dic)
            self.assertEqual(len(dic), 2)
            self.assertEqual(dic[u'key1'], u'val1')
            self.assertEqual(dic[u'2key'], u'value2')

        @mock.patch.object(subprocess, 'Popen')
        def test_findlibrary(self, mock_popen):
            if False:
                for i in range(10):
                    print('nop')
            'Test finding systemt libraries locations'
            mock_popen.side_effect = IOError
            libs = ['lib0', 'lib1', 'default_lib']
            res = _find_library(libs)
            self.assertEqual(res, libs[-1])
            libs = ['default_lib']
            res = _find_library(libs)
            self.assertEqual(res, libs[-1])
            mock_popen.side_effect = None

            class MockProcess(object):

                def communicate(self):
                    if False:
                        return 10
                    return ['a a a l\nb b b lb\nc c  c lib1\nd d d lib0\n']
            mock_popen.return_value = MockProcess()
            libs = ['lib0', 'lib1', 'default_lib']
            res = _find_library(libs)
            self.assertEqual(res, 'lib0')
if __name__ == '__main__':
    unittest.main()