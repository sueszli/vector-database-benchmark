import unittest
import os
import requests
import shutil
import tempfile
import secrets
import platform
import sys
from PySide6 import QtCore, QtTest, QtWidgets
from onionshare_cli.common import Common
from onionshare import Application, MainWindow, GuiCommon
from onionshare.tab.mode.share_mode import ShareMode
from onionshare.tab.mode.receive_mode import ReceiveMode
from onionshare.tab.mode.website_mode import WebsiteMode
from onionshare.tab.mode.chat_mode import ChatMode
from onionshare import strings

class GuiBaseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        common = sys.onionshare_common
        qtapp = sys.onionshare_qtapp
        shutil.rmtree(common.build_data_dir(), ignore_errors=True)
        common.gui = GuiCommon(common, qtapp, local_only=True)
        cls.gui = MainWindow(common, filenames=None)
        cls.gui.qtapp = qtapp
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.tmpfiles = []
        for _ in range(10):
            filename = os.path.join(cls.tmpdir.name, f'{secrets.token_hex(4)}.txt')
            with open(filename, 'w') as file:
                file.write(secrets.token_hex(10))
            cls.tmpfiles.append(filename)
        cls.tmpfile_test = os.path.join(cls.tmpdir.name, 'test.txt')
        with open(cls.tmpfile_test, 'w') as file:
            file.write('onionshare')
        cls.tmpfile_test2 = os.path.join(cls.tmpdir.name, 'test2.txt')
        with open(cls.tmpfile_test2, 'w') as file:
            file.write('onionshare2')
        cls.tmpfile_index_html = os.path.join(cls.tmpdir.name, 'index.html')
        with open(cls.tmpfile_index_html, 'w') as file:
            file.write('<html><body><p>This is a test website hosted by OnionShare</p></body></html>')
        cls.tmpfile_test_html = os.path.join(cls.tmpdir.name, 'test.html')
        with open(cls.tmpfile_test_html, 'w') as file:
            file.write('<html><body><p>Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?</p></body></html>')
        size = 1024 * 1024 * 155
        cls.tmpfile_large = os.path.join(cls.tmpdir.name, 'large_file')
        with open(cls.tmpfile_large, 'wb') as fout:
            fout.write(os.urandom(size))

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        cls.gui.qtapp.clipboard().clear()
        QtCore.QTimer.singleShot(200, cls.gui.close_dialog.accept_button.click)
        cls.gui.close()
        cls.gui.cleanup()

    def verify_new_tab(self, tab):
        if False:
            while True:
                i = 10
        QtTest.QTest.qWait(1000, self.gui.qtapp)
        self.assertTrue(tab.new_tab.isVisible())
        self.assertFalse(hasattr(tab, 'share_mode'))
        self.assertFalse(hasattr(tab, 'receive_mode'))
        self.assertFalse(hasattr(tab, 'website_mode'))
        self.assertFalse(hasattr(tab, 'chat_mode'))

    def new_share_tab(self):
        if False:
            i = 10
            return i + 15
        tab = self.gui.tabs.widget(0)
        self.verify_new_tab(tab)
        tab.share_button.click()
        self.assertFalse(tab.new_tab.isVisible())
        self.assertTrue(tab.share_mode.isVisible())
        return tab

    def new_share_tab_with_files(self):
        if False:
            i = 10
            return i + 15
        tab = self.new_share_tab()
        for filename in self.tmpfiles:
            tab.share_mode.server_status.file_selection.file_list.add_file(filename)
        return tab

    def new_receive_tab(self):
        if False:
            for i in range(10):
                print('nop')
        tab = self.gui.tabs.widget(0)
        self.verify_new_tab(tab)
        tab.receive_button.click()
        self.assertFalse(tab.new_tab.isVisible())
        self.assertTrue(tab.receive_mode.isVisible())
        return tab

    def new_website_tab(self):
        if False:
            for i in range(10):
                print('nop')
        tab = self.gui.tabs.widget(0)
        self.verify_new_tab(tab)
        tab.website_button.click()
        self.assertFalse(tab.new_tab.isVisible())
        self.assertTrue(tab.website_mode.isVisible())
        return tab

    def new_website_tab_with_files(self):
        if False:
            i = 10
            return i + 15
        tab = self.new_website_tab()
        for filename in self.tmpfiles:
            tab.website_mode.server_status.file_selection.file_list.add_file(filename)
        return tab

    def new_chat_tab(self):
        if False:
            i = 10
            return i + 15
        tab = self.gui.tabs.widget(0)
        self.verify_new_tab(tab)
        tab.chat_button.click()
        self.assertFalse(tab.new_tab.isVisible())
        self.assertTrue(tab.chat_mode.isVisible())
        return tab

    def close_all_tabs(self):
        if False:
            print('Hello World!')
        for _ in range(self.gui.tabs.count()):
            tab = self.gui.tabs.widget(0)
            QtCore.QTimer.singleShot(200, tab.close_dialog.accept_button.click)
            self.gui.tabs.tabBar().tabButton(0, QtWidgets.QTabBar.RightSide).click()

    def gui_loaded(self):
        if False:
            print('Hello World!')
        'Test that the GUI actually is shown'
        self.assertTrue(self.gui.show)

    def window_title_seen(self):
        if False:
            return 10
        'Test that the window title is OnionShare'
        self.assertEqual(self.gui.windowTitle(), 'OnionShare')

    def server_status_bar_is_visible(self):
        if False:
            return 10
        'Test that the status bar is visible'
        self.assertTrue(self.gui.status_bar.isVisible())

    def mode_settings_widget_is_visible(self, tab):
        if False:
            while True:
                i = 10
        'Test that the mode settings are visible'
        self.assertTrue(tab.get_mode().mode_settings_widget.isVisible())

    def mode_settings_widget_is_hidden(self, tab):
        if False:
            while True:
                i = 10
        'Test that the mode settings are hidden when the server starts'
        self.assertFalse(tab.get_mode().mode_settings_widget.isVisible())

    def click_toggle_history(self, tab):
        if False:
            print('Hello World!')
        'Test that we can toggle Download or Upload history by clicking the toggle button'
        currently_visible = tab.get_mode().history.isVisible()
        tab.get_mode().toggle_history.click()
        self.assertEqual(tab.get_mode().history.isVisible(), not currently_visible)

    def javascript_is_correct_mime_type(self, tab, file):
        if False:
            return 10
        'Test that the javascript file send.js is fetchable and that its MIME type is correct'
        path = f'{tab.get_mode().web.static_url_path}/js/{file}'
        url = f'http://127.0.0.1:{tab.app.port}/{path}'
        r = requests.get(url)
        self.assertTrue(r.headers['Content-Type'].startswith('text/javascript;'))

    def history_indicator(self, tab, indicator_count='1'):
        if False:
            while True:
                i = 10
        'Test that we can make sure the history is toggled off, do an action, and the indicator works'
        if tab.get_mode().history.isVisible():
            tab.get_mode().toggle_history.click()
            self.assertFalse(tab.get_mode().history.isVisible())
        self.assertFalse(tab.get_mode().toggle_history.indicator_label.isVisible())
        if type(tab.get_mode()) == ReceiveMode:
            files = {'file[]': open(self.tmpfiles[0], 'rb')}
            url = f'http://127.0.0.1:{tab.app.port}/upload'
            requests.post(url, files=files)
            QtTest.QTest.qWait(2000, self.gui.qtapp)
        if type(tab.get_mode()) == ShareMode:
            url = f'http://127.0.0.1:{tab.app.port}/download'
            requests.get(url)
            QtTest.QTest.qWait(2000, self.gui.qtapp)
        self.assertTrue(tab.get_mode().toggle_history.indicator_label.isVisible())
        self.assertEqual(tab.get_mode().toggle_history.indicator_label.text(), indicator_count)
        tab.get_mode().toggle_history.click()
        self.assertFalse(tab.get_mode().toggle_history.indicator_label.isVisible())

    def history_is_not_visible(self, tab):
        if False:
            return 10
        'Test that the History section is not visible'
        self.assertFalse(tab.get_mode().history.isVisible())

    def history_is_visible(self, tab):
        if False:
            return 10
        'Test that the History section is visible'
        self.assertTrue(tab.get_mode().history.isVisible())

    def server_working_on_start_button_pressed(self, tab):
        if False:
            i = 10
            return i + 15
        'Test we can start the service'
        tab.get_mode().server_status.server_button.click()
        self.assertEqual(tab.get_mode().server_status.status, 1)

    def toggle_indicator_is_reset(self, tab):
        if False:
            while True:
                i = 10
        self.assertEqual(tab.get_mode().toggle_history.indicator_count, 0)
        self.assertFalse(tab.get_mode().toggle_history.indicator_label.isVisible())

    def server_status_indicator_says_starting(self, tab):
        if False:
            while True:
                i = 10
        'Test that the Server Status indicator shows we are Starting'
        self.assertEqual(tab.get_mode().server_status_label.text(), strings._('gui_status_indicator_share_working'))

    def server_status_indicator_says_scheduled(self, tab):
        if False:
            while True:
                i = 10
        'Test that the Server Status indicator shows we are Scheduled'
        self.assertEqual(tab.get_mode().server_status_label.text(), strings._('gui_status_indicator_share_scheduled'))

    def server_is_started(self, tab, startup_time=2000):
        if False:
            i = 10
            return i + 15
        'Test that the server has started'
        QtTest.QTest.qWait(startup_time, self.gui.qtapp)
        self.assertEqual(tab.get_mode().server_status.status, 2)

    def web_server_is_running(self, tab):
        if False:
            for i in range(10):
                print('nop')
        'Test that the web server has started'
        try:
            requests.get(f'http://127.0.0.1:{tab.app.port}/')
            self.assertTrue(True)
        except requests.exceptions.ConnectionError:
            self.assertTrue(False)

    def add_button_visible(self, tab):
        if False:
            for i in range(10):
                print('nop')
        'Test that the add button should be visible'
        if platform.system() == 'Darwin':
            self.assertTrue(tab.get_mode().server_status.file_selection.add_files_button.isVisible())
            self.assertTrue(tab.get_mode().server_status.file_selection.add_folder_button.isVisible())
        else:
            self.assertTrue(tab.get_mode().server_status.file_selection.add_button.isVisible())

    def url_shown(self, tab):
        if False:
            print('Hello World!')
        'Test that the URL is showing'
        self.assertTrue(tab.get_mode().server_status.url.isVisible())

    def url_description_shown(self, tab):
        if False:
            print('Hello World!')
        'Test that the URL label is showing'
        self.assertTrue(tab.get_mode().server_status.url_description.isVisible())

    def url_instructions_shown(self, tab):
        if False:
            print('Hello World!')
        'Test that the URL instructions for sharing are showing'
        self.assertTrue(tab.get_mode().server_status.url_instructions.isVisible())

    def private_key_shown(self, tab):
        if False:
            for i in range(10):
                print('nop')
        'Test that the Private Key is showing when not in public mode'
        if not tab.settings.get('general', 'public'):
            self.assertTrue(tab.get_mode().server_status.private_key.isVisible())
            self.assertTrue(tab.get_mode().server_status.client_auth_toggle_button.isVisible())
            self.assertEqual(tab.get_mode().server_status.client_auth_toggle_button.text(), strings._('gui_reveal'))
            self.assertEqual(tab.get_mode().server_status.private_key.text(), '*' * len(tab.app.auth_string))
            tab.get_mode().server_status.client_auth_toggle_button.click()
            self.assertEqual(tab.get_mode().server_status.private_key.text(), tab.app.auth_string)
            self.assertEqual(tab.get_mode().server_status.client_auth_toggle_button.text(), strings._('gui_hide'))
            tab.get_mode().server_status.client_auth_toggle_button.click()
            self.assertEqual(tab.get_mode().server_status.private_key.text(), '*' * len(tab.app.auth_string))
            self.assertEqual(tab.get_mode().server_status.client_auth_toggle_button.text(), strings._('gui_reveal'))
        else:
            self.assertFalse(tab.get_mode().server_status.private_key.isVisible())

    def client_auth_instructions_shown(self, tab):
        if False:
            return 10
        '\n        Test that the Private Key instructions for sharing\n        are showing when not in public mode\n        '
        if not tab.settings.get('general', 'public'):
            self.assertTrue(tab.get_mode().server_status.client_auth_instructions.isVisible())
        else:
            self.assertFalse(tab.get_mode().server_status.client_auth_instructions.isVisible())

    def have_copy_url_button(self, tab):
        if False:
            print('Hello World!')
        'Test that the Copy URL button is shown and that the clipboard is correct'
        self.assertTrue(tab.get_mode().server_status.copy_url_button.isVisible())
        tab.get_mode().server_status.copy_url_button.click()
        clipboard = tab.common.gui.qtapp.clipboard()
        self.assertEqual(clipboard.text(), f'http://127.0.0.1:{tab.app.port}')

    def have_show_url_qr_code_button(self, tab):
        if False:
            while True:
                i = 10
        'Test that the Show QR Code URL button is shown and that it loads a QR Code Dialog'
        self.assertTrue(tab.get_mode().server_status.show_url_qr_code_button.isVisible())

        def accept_dialog():
            if False:
                i = 10
                return i + 15
            window = tab.common.gui.qtapp.activeWindow()
            if window:
                window.close()
        QtCore.QTimer.singleShot(500, accept_dialog)
        tab.get_mode().server_status.show_url_qr_code_button.click()

    def have_show_client_auth_qr_code_button(self, tab):
        if False:
            while True:
                i = 10
        '\n        Test that the Show QR Code Client Auth button is shown when\n        not in public mode and that it loads a QR Code Dialog.\n        '
        if not tab.settings.get('general', 'public'):
            self.assertTrue(tab.get_mode().server_status.show_client_auth_qr_code_button.isVisible())

            def accept_dialog():
                if False:
                    for i in range(10):
                        print('nop')
                window = tab.common.gui.qtapp.activeWindow()
                if window:
                    window.close()
            QtCore.QTimer.singleShot(500, accept_dialog)
            tab.get_mode().server_status.show_client_auth_qr_code_button.click()
        else:
            self.assertFalse(tab.get_mode().server_status.show_client_auth_qr_code_button.isVisible())

    def server_status_indicator_says_started(self, tab):
        if False:
            while True:
                i = 10
        'Test that the Server Status indicator shows we are started'
        if type(tab.get_mode()) == ReceiveMode:
            self.assertEqual(tab.get_mode().server_status_label.text(), strings._('gui_status_indicator_receive_started'))
        if type(tab.get_mode()) == ShareMode:
            self.assertEqual(tab.get_mode().server_status_label.text(), strings._('gui_status_indicator_share_started'))

    def web_page(self, tab, string):
        if False:
            print('Hello World!')
        'Test that the web page contains a string'
        url = f'http://127.0.0.1:{tab.app.port}/'
        r = requests.get(url)
        self.assertTrue(string in r.text)

    def history_widgets_present(self, tab):
        if False:
            return 10
        'Test that the relevant widgets are present in the history view after activity has taken place'
        self.assertFalse(tab.get_mode().history.empty.isVisible())
        self.assertTrue(tab.get_mode().history.not_empty.isVisible())

    def counter_incremented(self, tab, count):
        if False:
            return 10
        'Test that the counter has incremented'
        self.assertEqual(tab.get_mode().history.completed_count, count)

    def server_is_stopped(self, tab):
        if False:
            while True:
                i = 10
        'Test that the server stops when we click Stop'
        if type(tab.get_mode()) == ReceiveMode or (type(tab.get_mode()) == ShareMode and (not tab.settings.get('share', 'autostop_sharing'))) or type(tab.get_mode()) == WebsiteMode or (type(tab.get_mode()) == ChatMode):
            tab.get_mode().server_status.server_button.click()
        self.assertEqual(tab.get_mode().server_status.status, 0)
        self.assertFalse(tab.get_mode().server_status.show_url_qr_code_button.isVisible())
        self.assertFalse(tab.get_mode().server_status.copy_url_button.isVisible())
        self.assertFalse(tab.get_mode().server_status.url.isVisible())
        self.assertFalse(tab.get_mode().server_status.url_description.isVisible())
        self.assertFalse(tab.get_mode().server_status.url_instructions.isVisible())
        self.assertFalse(tab.get_mode().server_status.private_key.isVisible())
        self.assertFalse(tab.get_mode().server_status.client_auth_instructions.isVisible())
        self.assertFalse(tab.get_mode().server_status.copy_client_auth_button.isVisible())

    def web_server_is_stopped(self, tab):
        if False:
            while True:
                i = 10
        'Test that the web server also stopped'
        QtTest.QTest.qWait(800, self.gui.qtapp)
        try:
            requests.get(f'http://127.0.0.1:{tab.app.port}/')
            self.assertTrue(False)
        except requests.exceptions.ConnectionError:
            self.assertTrue(True)

    def server_status_indicator_says_closed(self, tab):
        if False:
            while True:
                i = 10
        'Test that the Server Status indicator shows we closed'
        if type(tab.get_mode()) == ReceiveMode:
            self.assertEqual(tab.get_mode().server_status_label.text(), strings._('gui_status_indicator_receive_stopped'))
        if type(tab.get_mode()) == ShareMode:
            if not tab.settings.get('share', 'autostop_sharing'):
                self.assertEqual(tab.get_mode().server_status_label.text(), strings._('gui_status_indicator_share_stopped'))
            else:
                self.assertEqual(tab.get_mode().server_status_label.text(), strings._('closing_automatically'))

    def clear_all_history_items(self, tab, count):
        if False:
            i = 10
            return i + 15
        if count == 0:
            tab.get_mode().history.clear_button.click()
        self.assertEqual(len(tab.get_mode().history.item_list.items.keys()), count)

    def file_selection_widget_has_files(self, tab, num=3):
        if False:
            while True:
                i = 10
        'Test that the number of items in the list is as expected'
        self.assertEqual(tab.get_mode().server_status.file_selection.get_num_files(), num)

    def add_remove_buttons_hidden(self, tab):
        if False:
            i = 10
            return i + 15
        'Test that the add and remove buttons are hidden when the server starts'
        if platform.system() == 'Darwin':
            self.assertFalse(tab.get_mode().server_status.file_selection.add_files_button.isVisible())
            self.assertFalse(tab.get_mode().server_status.file_selection.add_folder_button.isVisible())
        else:
            self.assertFalse(tab.get_mode().server_status.file_selection.add_button.isVisible())
        self.assertFalse(tab.get_mode().server_status.file_selection.remove_button.isVisible())

    def set_timeout(self, tab, timeout):
        if False:
            return 10
        'Test that the timeout can be set'
        timer = QtCore.QDateTime.currentDateTime().addSecs(timeout)
        tab.get_mode().mode_settings_widget.autostop_timer_widget.setDateTime(timer)
        self.assertTrue(tab.get_mode().mode_settings_widget.autostop_timer_widget.dateTime(), timer)

    def autostop_timer_widget_hidden(self, tab):
        if False:
            for i in range(10):
                print('nop')
        'Test that the auto-stop timer widget is hidden when share has started'
        self.assertFalse(tab.get_mode().mode_settings_widget.autostop_timer_widget.isVisible())

    def server_timed_out(self, tab, wait):
        if False:
            for i in range(10):
                print('nop')
        'Test that the server has timed out after the timer ran out'
        QtTest.QTest.qWait(wait, self.gui.qtapp)
        self.assertEqual(tab.get_mode().server_status.status, 0)

    def clientauth_is_visible(self, tab):
        if False:
            return 10
        'Test that the ClientAuth button is visible and that the clipboard contains its contents'
        self.assertTrue(tab.get_mode().server_status.copy_client_auth_button.isVisible())
        tab.get_mode().server_status.copy_client_auth_button.click()
        clipboard = tab.common.gui.qtapp.clipboard()
        self.assertEqual(clipboard.text(), 'E2GOT5LTUTP3OAMRCRXO4GSH6VKJEUOXZQUC336SRKAHTTT5OVSA')

    def clientauth_is_not_visible(self, tab):
        if False:
            return 10
        'Test that the ClientAuth button is not visible'
        self.assertFalse(tab.get_mode().server_status.copy_client_auth_button.isVisible())

    def hit_405(self, url, expected_resp, data={}, methods=[]):
        if False:
            return 10
        'Test various HTTP methods and the response'
        for method in methods:
            if method == 'put':
                r = requests.put(url, data=data)
            if method == 'post':
                r = requests.post(url, data=data)
            if method == 'delete':
                r = requests.delete(url)
            if method == 'options':
                r = requests.options(url)
            self.assertTrue(expected_resp in r.text)
            self.assertFalse('Werkzeug' in r.headers)

    def run_all_common_setup_tests(self):
        if False:
            i = 10
            return i + 15
        self.gui_loaded()
        self.window_title_seen()
        self.server_status_bar_is_visible()