"""Tests for the Existing Kernel Connection widget."""
import sys
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialogButtonBox
from spyder.config.base import running_in_ci
from spyder.plugins.ipythonconsole.widgets.kernelconnect import KernelConnectionDialog
from spyder.config.manager import CONF

@pytest.fixture
def connection_dialog_factory(qtbot, request):
    if False:
        print('Hello World!')
    'Set up kernel connection dialog.'

    class DialogFactory(object):

        def get_default_dialog(self):
            if False:
                return 10
            dialog = KernelConnectionDialog()
            request.addfinalizer(dialog.close)
            return dialog

        def submit_filled_dialog(self, use_keyfile, save_settings):
            if False:
                for i in range(10):
                    print('nop')
            dlg = self.get_default_dialog()
            dlg.cf.clear()
            qtbot.keyClicks(dlg.cf, pytest.cf_path)
            dlg.rm_group.setChecked(True)
            dlg.hn.clear()
            qtbot.keyClicks(dlg.hn, pytest.hn)
            dlg.pn.clear()
            qtbot.keyClicks(dlg.pn, str(pytest.pn))
            dlg.un.clear()
            qtbot.keyClicks(dlg.un, pytest.un)
            if use_keyfile:
                dlg.kf_radio.setChecked(True)
                assert dlg.kf.isEnabled()
                dlg.kf.clear()
                qtbot.keyClicks(dlg.kf, pytest.kf)
                dlg.kfp.clear()
                qtbot.keyClicks(dlg.kfp, pytest.kfp)
            else:
                dlg.pw_radio.setChecked(True)
                assert dlg.pw.isEnabled()
                dlg.pw.clear()
                qtbot.keyClicks(dlg.pw, pytest.pw)
            dlg.save_layout.setChecked(save_settings)
            return dlg

    def teardown():
        if False:
            for i in range(10):
                print('nop')
        'Clear existing-kernel config and keyring passwords.'
        CONF.remove_section('existing-kernel')
        try:
            import keyring
            keyring.set_password('spyder_remote_kernel', 'ssh_key_passphrase', '')
            keyring.set_password('spyder_remote_kernel', 'ssh_password', '')
        except Exception:
            pass
    pytest.cf_path = 'cf_path'
    pytest.un = 'test_username'
    pytest.hn = 'test_hostname'
    pytest.pn = 123
    pytest.kf = 'test_kf'
    pytest.kfp = 'test_kfp'
    pytest.pw = 'test_pw'
    request.addfinalizer(teardown)
    return DialogFactory()

def test_connection_dialog_remembers_input_with_ssh_passphrase(qtbot, connection_dialog_factory):
    if False:
        while True:
            i = 10
    "\n    Test that the dialog remembers the user's kernel connection\n    settings and ssh key passphrase when the user checks the\n    save checkbox.\n    "
    dlg = connection_dialog_factory.submit_filled_dialog(use_keyfile=True, save_settings=True)
    qtbot.mouseClick(dlg.accept_btns.button(QDialogButtonBox.Ok), Qt.LeftButton)
    new_dlg = connection_dialog_factory.get_default_dialog()
    assert new_dlg.cf.text() == pytest.cf_path
    assert new_dlg.rm_group.isChecked()
    assert new_dlg.hn.text() == pytest.hn
    assert new_dlg.un.text() == pytest.un
    assert new_dlg.pn.text() == str(pytest.pn)
    assert new_dlg.kf.text() == pytest.kf
    if not running_in_ci():
        assert new_dlg.kfp.text() == pytest.kfp

def test_connection_dialog_doesnt_remember_input_with_ssh_passphrase(qtbot, connection_dialog_factory):
    if False:
        for i in range(10):
            print('nop')
    "\n    Test that the dialog doesn't remember the user's kernel\n    connection settings and ssh key passphrase when the user doesn't\n    check the save checkbox.\n    "
    dlg = connection_dialog_factory.submit_filled_dialog(use_keyfile=True, save_settings=False)
    qtbot.mouseClick(dlg.accept_btns.button(QDialogButtonBox.Ok), Qt.LeftButton)
    new_dlg = connection_dialog_factory.get_default_dialog()
    assert new_dlg.cf.text() == ''
    assert not new_dlg.rm_group.isChecked()
    assert new_dlg.hn.text() == ''
    assert new_dlg.un.text() == ''
    assert new_dlg.pn.text() == '22'
    assert new_dlg.kf.text() == ''
    if not running_in_ci():
        assert new_dlg.kfp.text() == ''

def test_connection_dialog_remembers_input_with_password(qtbot, connection_dialog_factory):
    if False:
        i = 10
        return i + 15
    "\n    Test that the dialog remembers the user's kernel connection\n    settings and ssh password when the user checks the save checkbox.\n    "
    dlg = connection_dialog_factory.submit_filled_dialog(use_keyfile=False, save_settings=True)
    qtbot.mouseClick(dlg.accept_btns.button(QDialogButtonBox.Ok), Qt.LeftButton)
    new_dlg = connection_dialog_factory.get_default_dialog()
    assert new_dlg.cf.text() == pytest.cf_path
    assert new_dlg.rm_group.isChecked()
    assert new_dlg.hn.text() == pytest.hn
    assert new_dlg.un.text() == pytest.un
    assert new_dlg.pn.text() == str(pytest.pn)
    if not running_in_ci():
        assert new_dlg.pw.text() == pytest.pw

def test_connection_dialog_doesnt_remember_input_with_password(qtbot, connection_dialog_factory):
    if False:
        print('Hello World!')
    "\n    Test that the dialog doesn't remember the user's kernel\n    connection settings and ssh password when the user doesn't\n    check the save checkbox.\n    "
    dlg = connection_dialog_factory.submit_filled_dialog(use_keyfile=False, save_settings=False)
    qtbot.mouseClick(dlg.accept_btns.button(QDialogButtonBox.Ok), Qt.LeftButton)
    new_dlg = connection_dialog_factory.get_default_dialog()
    assert new_dlg.cf.text() == ''
    assert not new_dlg.rm_group.isChecked()
    assert new_dlg.hn.text() == ''
    assert new_dlg.un.text() == ''
    assert new_dlg.pn.text() == '22'
    if not running_in_ci():
        assert new_dlg.pw.text() == ''
if __name__ == '__main__':
    pytest.main()