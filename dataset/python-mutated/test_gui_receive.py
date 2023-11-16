import glob
import pytest
import os
import requests
import shutil
import sys
from datetime import datetime, timedelta
from PySide6 import QtCore, QtTest
from .gui_base_test import GuiBaseTest

class TestReceive(GuiBaseTest):

    def upload_file(self, tab, file_to_upload, expected_basename, identical_files_at_once=False):
        if False:
            return 10
        'Test that we can upload the file'
        QtTest.QTest.qWait(2000, self.gui.qtapp)
        files = {'file[]': open(file_to_upload, 'rb')}
        url = f'http://127.0.0.1:{tab.app.port}/upload'
        requests.post(url, files=files)
        if identical_files_at_once:
            requests.post(url, files=files)
        QtTest.QTest.qWait(1000, self.gui.qtapp)
        exists = False
        now = datetime.now()
        for _ in range(10):
            date_dir = now.strftime('%Y-%m-%d')
            time_dir = now.strftime('%H%M%S')
            receive_mode_dir = os.path.join(tab.settings.get('receive', 'data_dir'), date_dir, time_dir)
            for path in glob.glob(receive_mode_dir + '*'):
                if os.path.exists(os.path.join(path, expected_basename)):
                    exists = True
                    break
            now = now - timedelta(seconds=1)
        self.assertTrue(exists)

    def upload_file_should_fail(self, tab):
        if False:
            while True:
                i = 10
        "Test that we can't upload the file when permissions are wrong, and expected content is shown"
        QtTest.QTest.qWait(1000, self.gui.qtapp)
        files = {'file[]': open(self.tmpfile_test, 'rb')}
        url = f'http://127.0.0.1:{tab.app.port}/upload'
        r = requests.post(url, files=files)

        def accept_dialog():
            if False:
                print('Hello World!')
            window = tab.common.gui.qtapp.activeWindow()
            if window:
                window.close()
        QtCore.QTimer.singleShot(1000, accept_dialog)
        self.assertTrue('Error uploading, please inform the OnionShare user' in r.text)

    def submit_message(self, tab, message):
        if False:
            while True:
                i = 10
        'Test that we can submit a message'
        QtTest.QTest.qWait(2000, self.gui.qtapp)
        url = f'http://127.0.0.1:{tab.app.port}/upload'
        requests.post(url, data={'text': message})
        QtTest.QTest.qWait(1000, self.gui.qtapp)
        exists = False
        now = datetime.now()
        for _ in range(10):
            date_dir = now.strftime('%Y-%m-%d')
            time_dir = now.strftime('%H%M%S')
            expected_estimated_filename = os.path.join(tab.settings.get('receive', 'data_dir'), date_dir, f'{time_dir}*-message.txt')
            for path in glob.glob(expected_estimated_filename):
                if os.path.exists(path):
                    with open(path) as f:
                        assert f.read() == message
                    exists = True
                    break
            now = now - timedelta(seconds=1)
        self.assertTrue(exists)

    def run_all_receive_mode_setup_tests(self, tab):
        if False:
            print('Hello World!')
        'Set up a share in Receive mode and start it'
        self.history_is_not_visible(tab)
        self.click_toggle_history(tab)
        self.history_is_visible(tab)
        self.server_working_on_start_button_pressed(tab)
        self.server_status_indicator_says_starting(tab)
        self.server_is_started(tab)
        self.web_server_is_running(tab)
        self.url_description_shown(tab)
        self.url_instructions_shown(tab)
        self.url_shown(tab)
        self.have_copy_url_button(tab)
        self.have_show_url_qr_code_button(tab)
        self.client_auth_instructions_shown(tab)
        self.private_key_shown(tab)
        self.have_show_client_auth_qr_code_button(tab)
        self.server_status_indicator_says_started(tab)

    def run_all_receive_mode_tests(self, tab):
        if False:
            print('Hello World!')
        'Submit files and messages in receive mode and stop the share'
        self.run_all_receive_mode_setup_tests(tab)
        self.javascript_is_correct_mime_type(tab, 'receive.js')
        self.upload_file(tab, self.tmpfile_test, 'test.txt')
        self.history_widgets_present(tab)
        self.counter_incremented(tab, 1)
        self.upload_file(tab, self.tmpfile_test, 'test.txt')
        self.counter_incremented(tab, 2)
        self.upload_file(tab, self.tmpfile_test2, 'test2.txt')
        self.counter_incremented(tab, 3)
        self.upload_file(tab, self.tmpfile_test2, 'test2.txt')
        self.counter_incremented(tab, 4)
        self.submit_message(tab, 'onionshare is an interesting piece of software')
        self.counter_incremented(tab, 5)
        self.upload_file(tab, self.tmpfile_test, 'test.txt', True)
        self.counter_incremented(tab, 7)
        self.history_indicator(tab, '2')
        self.server_is_stopped(tab)
        self.web_server_is_stopped(tab)
        self.server_status_indicator_says_closed(tab)
        self.server_working_on_start_button_pressed(tab)
        self.server_is_started(tab)
        self.history_indicator(tab, '2')

    def run_all_clear_all_button_tests(self, tab):
        if False:
            for i in range(10):
                print('nop')
        'Test the Clear All history button'
        self.run_all_receive_mode_setup_tests(tab)
        self.upload_file(tab, self.tmpfile_test, 'test.txt')
        self.history_widgets_present(tab)
        self.clear_all_history_items(tab, 0)
        self.upload_file(tab, self.tmpfile_test, 'test.txt')
        self.clear_all_history_items(tab, 2)

    def run_all_upload_non_writable_dir_tests(self, tab):
        if False:
            for i in range(10):
                print('nop')
        'Test uploading a file when the data_dir is non-writable'
        upload_dir = os.path.join(self.tmpdir.name, 'OnionShare')
        shutil.rmtree(upload_dir, ignore_errors=True)
        os.makedirs(upload_dir, 448)
        tab.get_mode().data_dir_lineedit.setText(upload_dir)
        tab.settings.set('receive', 'data_dir', upload_dir)
        self.run_all_receive_mode_setup_tests(tab)
        os.chmod(upload_dir, 256)
        self.upload_file_should_fail(tab)
        self.server_is_stopped(tab)
        self.web_server_is_stopped(tab)
        self.server_status_indicator_says_closed(tab)
        os.chmod(upload_dir, 448)

    def test_clear_all_button(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Clear all history items should work\n        '
        tab = self.new_receive_tab()
        self.run_all_common_setup_tests()
        self.run_all_clear_all_button_tests(tab)
        self.close_all_tabs()

    def test_autostop_timer(self):
        if False:
            while True:
                i = 10
        '\n        Test autostop timer\n        '
        tab = self.new_receive_tab()
        tab.get_mode().mode_settings_widget.toggle_advanced_button.click()
        tab.get_mode().mode_settings_widget.autostop_timer_checkbox.click()
        self.run_all_common_setup_tests()
        self.run_all_receive_mode_setup_tests(tab)
        self.set_timeout(tab, 5)
        self.autostop_timer_widget_hidden(tab)
        self.server_timed_out(tab, 15000)
        self.web_server_is_stopped(tab)
        self.close_all_tabs()

    def test_upload(self):
        if False:
            while True:
                i = 10
        '\n        Test uploading files\n        '
        tab = self.new_receive_tab()
        self.run_all_common_setup_tests()
        self.run_all_receive_mode_tests(tab)
        self.close_all_tabs()

    @pytest.mark.skipif(sys.platform == 'win32', reason="Windows doesn't have chmod")
    def test_upload_non_writable_dir(self):
        if False:
            print('Hello World!')
        '\n        Test uploading files to a non-writable directory\n        '
        tab = self.new_receive_tab()
        self.run_all_upload_non_writable_dir_tests(tab)
        self.close_all_tabs()

    def test_public_upload(self):
        if False:
            i = 10
            return i + 15
        '\n        Test uploading files in public mode\n        '
        tab = self.new_receive_tab()
        tab.get_mode().mode_settings_widget.public_checkbox.click()
        self.run_all_common_setup_tests()
        self.run_all_receive_mode_tests(tab)
        self.close_all_tabs()

    @pytest.mark.skipif(sys.platform == 'win32', reason="Windows doesn't have chmod")
    def test_public_upload_non_writable_dir(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test uploading files to a non-writable directory in public mode\n        '
        tab = self.new_receive_tab()
        tab.get_mode().mode_settings_widget.public_checkbox.click()
        self.run_all_upload_non_writable_dir_tests(tab)
        self.close_all_tabs()

    def test_405_page_returned_for_invalid_methods(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Our custom 405 page should return for invalid methods\n        '
        tab = self.new_receive_tab()
        tab.get_mode().mode_settings_widget.public_checkbox.click()
        self.run_all_common_setup_tests()
        self.run_all_receive_mode_setup_tests(tab)
        self.upload_file(tab, self.tmpfile_test, 'test.txt')
        url = f'http://127.0.0.1:{tab.app.port}/'
        self.hit_405(url, expected_resp='OnionShare: 405 Method Not Allowed', data={'foo': 'bar'}, methods=['put', 'post', 'delete', 'options'])
        self.server_is_stopped(tab)
        self.web_server_is_stopped(tab)
        self.server_status_indicator_says_closed(tab)
        self.close_all_tabs()