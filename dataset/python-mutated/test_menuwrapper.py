"""Tests for Menu"""
import sys
import os
import unittest
sys.path.append('.')
from pywinauto.windows.application import Application
from pywinauto.sysinfo import is_x64_Python
from pywinauto.controls.menuwrapper import MenuItemNotEnabled
from pywinauto.timings import Timings
mfc_samples_folder = os.path.join(os.path.dirname(__file__), '..\\..\\apps\\MFC_samples')
if is_x64_Python():
    mfc_samples_folder = os.path.join(mfc_samples_folder, 'x64')

class MenuWrapperTests(unittest.TestCase):
    """Unit tests for the Menu and the MenuItem classes"""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Set some data and ensure the application is in the state we want'
        Timings.defaults()
        self.app = Application()
        self.app.start('Notepad.exe')
        self.dlg = self.app.Notepad

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        'Close the application after tests'
        self.app.kill()

    def testInvalidHandle(self):
        if False:
            return 10
        'Test that an exception is raised with an invalid menu handle'
        pass

    def testItemCount(self):
        if False:
            while True:
                i = 10
        self.assertEqual(5, self.dlg.menu().item_count())

    def testItem(self):
        if False:
            while True:
                i = 10
        self.assertEqual(u'&File', self.dlg.menu().item(0).text())
        self.assertEqual(u'&File', self.dlg.menu().item(u'File').text())
        self.assertEqual(u'&File', self.dlg.menu().item(u'&File', exact=True).text())

    def testItems(self):
        if False:
            print('Hello World!')
        self.assertEqual([u'&File', u'&Edit', u'F&ormat', u'&View', u'&Help'], [item.text() for item in self.dlg.menu().items()])

    def testFriendlyClassName(self):
        if False:
            return 10
        self.assertEqual('MenuItem', self.dlg.menu().item(0).friendly_class_name())

    def testMenuItemNotEnabled(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(MenuItemNotEnabled, self.dlg.menu_select, 'Edit->Find Next')
        self.assertRaises(MenuItemNotEnabled, self.dlg.menu_item('Edit->Find Next').click)
        self.assertRaises(MenuItemNotEnabled, self.dlg.menu_item('Edit->Find Next').click_input)

    def testGetProperties(self):
        if False:
            print('Hello World!')
        self.assertEqual({u'menu_items': [{u'index': 0, u'state': 0, u'item_type': 0, u'item_id': 64, u'text': u'View &Help'}, {u'index': 1, u'state': 3, u'item_type': 2048, u'item_id': 0, u'text': u''}, {u'index': 2, u'state': 0, u'item_type': 0, u'item_id': 65, u'text': u'&About Notepad'}]}, self.dlg.menu().get_menu_path('Help')[0].sub_menu().get_properties())

    def testGetMenuPath(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(u'&About Notepad', self.dlg.menu().get_menu_path(' Help -> #2 ')[-1].text())
        self.assertEqual(u'&About Notepad', self.dlg.menu().get_menu_path('Help->$65')[-1].text())
        self.assertEqual(u'&About Notepad', self.dlg.menu().get_menu_path('&Help->&About Notepad', exact=True)[-1].text())
        self.assertRaises(IndexError, self.dlg.menu().get_menu_path, '&Help->About what?', exact=True)

    def test__repr__(self):
        if False:
            print('Hello World!')
        print(self.dlg.menu())
        print(self.dlg.menu().get_menu_path('&Help->&About Notepad', exact=True)[-1])

    def testClick(self):
        if False:
            while True:
                i = 10
        self.dlg.menu().get_menu_path('&Help->&About Notepad')[-1].click()
        About = self.app.window(name='About Notepad')
        About.wait('ready')
        About.OK.click()
        About.wait_not('visible')

    def testClickInput(self):
        if False:
            return 10
        self.dlg.menu().get_menu_path('&Help->&About Notepad')[-1].click_input()
        About = self.app.window(name='About Notepad')
        About.wait('ready')
        About.OK.click()
        About.wait_not('visible')

class OwnerDrawnMenuTests(unittest.TestCase):
    """Unit tests for the OWNERDRAW menu items"""

    def setUp(self):
        if False:
            print('Hello World!')
        'Set some data and ensure the application is in the state we want'
        Timings.defaults()
        self.app = Application().start(os.path.join(mfc_samples_folder, u'BCDialogMenu.exe'))
        self.dlg = self.app.BCDialogMenu
        self.app.wait_cpu_usage_lower(threshold=1.5, timeout=30, usage_interval=1)
        self.dlg.wait('ready')

    def tearDown(self):
        if False:
            print('Hello World!')
        'Close the application after tests'
        self.app.kill()

    def testCorrectText(self):
        if False:
            return 10
        menu = self.dlg.menu()
        self.assertEqual(u'&New', menu.get_menu_path('&File->#0')[-1].text()[:4])
        self.assertEqual(u'&Open...', menu.get_menu_path('&File->#1')[-1].text()[:8])
if __name__ == '__main__':
    unittest.main()