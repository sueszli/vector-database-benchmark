"""
This file tests the case that PTB was installed *without* the optional dependency `passport`.
Currently this only means that cryptography is not installed.

Because imports in pytest are intricate, we just run
    pytest -k test_no_passport.py

with the TEST_WITH_OPT_DEPS environment variable set to False in addition to the regular test suite
"""
import pytest
from telegram import _bot as bot
from telegram._passport import credentials
from tests.auxil.envvars import TEST_WITH_OPT_DEPS

@pytest.mark.skipif(TEST_WITH_OPT_DEPS, reason='Only relevant if the optional dependency is not installed')
class TestNoPassportWithoutRequest:

    def test_bot_init(self, bot_info):
        if False:
            return 10
        with pytest.raises(RuntimeError, match='passport'):
            bot.Bot(bot_info['token'], private_key=1, private_key_password=2)

    def test_credentials_decrypt(self):
        if False:
            return 10
        with pytest.raises(RuntimeError, match='passport'):
            credentials.decrypt(1, 1, 1)

    def test_encrypted_credentials_decrypted_secret(self):
        if False:
            return 10
        ec = credentials.EncryptedCredentials('data', 'hash', 'secret')
        with pytest.raises(RuntimeError, match='passport'):
            ec.decrypted_secret