"""Unit tests for Keychain API."""
from __future__ import print_function, unicode_literals
import pytest
from workflow.workflow import PasswordNotFound, KeychainError
from .conftest import BUNDLE_ID
ACCOUNT = 'this-is-my-test-account'
PASSWORD = 'hunter2'
PASSWORD2 = 'hunter2ing'
PASSWORD3 = 'hünter\\“2”'

def test_keychain(wf):
    if False:
        for i in range(10):
            print('nop')
    'Save/get/delete password'
    try:
        wf.delete_password(ACCOUNT)
    except PasswordNotFound:
        pass
    with pytest.raises(PasswordNotFound):
        wf.delete_password(ACCOUNT)
    with pytest.raises(PasswordNotFound):
        wf.get_password(ACCOUNT)
    wf.save_password(ACCOUNT, PASSWORD)
    assert wf.get_password(ACCOUNT) == PASSWORD
    assert wf.get_password(ACCOUNT, BUNDLE_ID)
    wf.save_password(ACCOUNT, PASSWORD)
    assert wf.get_password(ACCOUNT) == PASSWORD
    wf.save_password(ACCOUNT, PASSWORD2)
    assert wf.get_password(ACCOUNT) == PASSWORD2
    wf.save_password(ACCOUNT, PASSWORD3)
    assert wf.get_password(ACCOUNT) == PASSWORD3
    with pytest.raises(KeychainError):
        wf._call_security('pants', BUNDLE_ID, ACCOUNT)