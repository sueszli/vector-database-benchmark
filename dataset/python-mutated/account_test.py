"""Tests for certbot._internal.account."""
import datetime
import json
import sys
import unittest
from unittest import mock
import josepy as jose
import pytest
import pytz
from acme import messages
from certbot import errors
from certbot.compat import filesystem
from certbot.compat import misc
from certbot.compat import os
import certbot.tests.util as test_util
KEY = jose.JWKRSA.load(test_util.load_vector('rsa512_key.pem'))

class AccountTest(unittest.TestCase):
    """Tests for certbot._internal.account.Account."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        from certbot._internal.account import Account
        self.regr = mock.MagicMock()
        self.meta = Account.Meta(creation_host='test.certbot.org', creation_dt=datetime.datetime(2015, 7, 4, 14, 4, 10, tzinfo=pytz.UTC))
        self.acc = Account(self.regr, KEY, self.meta)
        self.regr.__repr__ = mock.MagicMock(return_value='i_am_a_regr')
        with mock.patch('certbot._internal.account.socket') as mock_socket:
            mock_socket.getfqdn.return_value = 'test.certbot.org'
            with mock.patch('certbot._internal.account.datetime') as mock_dt:
                mock_dt.datetime.now.return_value = self.meta.creation_dt
                self.acc_no_meta = Account(self.regr, KEY)

    def test_init(self):
        if False:
            return 10
        assert self.regr == self.acc.regr
        assert KEY == self.acc.key
        assert self.meta == self.acc_no_meta.meta

    def test_id(self):
        if False:
            i = 10
            return i + 15
        assert self.acc.id == '7adac10320f585ddf118429c0c4af2cd'

    def test_slug(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.acc.slug == 'test.certbot.org@2015-07-04T14:04:10Z (7ada)'

    def test_repr(self):
        if False:
            i = 10
            return i + 15
        assert repr(self.acc).startswith('<Account(i_am_a_regr, 7adac10320f585ddf118429c0c4af2cd, Meta(')

class MetaTest(unittest.TestCase):
    """Tests for certbot._internal.account.Meta."""

    def test_deserialize_partial(self):
        if False:
            while True:
                i = 10
        from certbot._internal.account import Account
        meta = Account.Meta.json_loads('{   "creation_dt": "2020-06-13T07:46:45Z",   "creation_host": "hyperion.localdomain"}')
        assert meta.creation_dt is not None
        assert meta.creation_host is not None
        assert meta.register_to_eff is None

    def test_deserialize_full(self):
        if False:
            return 10
        from certbot._internal.account import Account
        meta = Account.Meta.json_loads('{   "creation_dt": "2020-06-13T07:46:45Z",   "creation_host": "hyperion.localdomain",   "register_to_eff": "bar"}')
        assert meta.creation_dt is not None
        assert meta.creation_host is not None
        assert meta.register_to_eff is not None

class AccountMemoryStorageTest(unittest.TestCase):
    """Tests for certbot._internal.account.AccountMemoryStorage."""

    def setUp(self):
        if False:
            while True:
                i = 10
        from certbot._internal.account import AccountMemoryStorage
        self.storage = AccountMemoryStorage()

    def test_it(self):
        if False:
            while True:
                i = 10
        account = mock.Mock(id='x')
        assert [] == self.storage.find_all()
        with pytest.raises(errors.AccountNotFound):
            self.storage.load('x')
        self.storage.save(account, None)
        assert [account] == self.storage.find_all()
        assert account == self.storage.load('x')
        self.storage.save(account, None)
        assert [account] == self.storage.find_all()

class AccountFileStorageTest(test_util.ConfigTestCase):
    """Tests for certbot._internal.account.AccountFileStorage."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        from certbot._internal.account import AccountFileStorage
        self.storage = AccountFileStorage(self.config)
        from certbot._internal.account import Account
        meta = Account.Meta(creation_host='test.example.org', creation_dt=datetime.datetime(2021, 1, 5, 14, 4, 10, tzinfo=pytz.UTC))
        self.acc = Account(regr=messages.RegistrationResource(uri=None, body=messages.Registration()), key=KEY, meta=meta)
        self.mock_client = mock.MagicMock()

    def test_init_creates_dir(self):
        if False:
            i = 10
            return i + 15
        assert os.path.isdir(misc.underscores_for_unsupported_characters_in_path(self.config.accounts_dir))

    def test_save_and_restore(self):
        if False:
            while True:
                i = 10
        self.storage.save(self.acc, self.mock_client)
        account_path = os.path.join(self.config.accounts_dir, self.acc.id)
        assert os.path.exists(account_path)
        for file_name in ('regr.json', 'meta.json', 'private_key.json'):
            assert os.path.exists(os.path.join(account_path, file_name))
        assert filesystem.check_mode(os.path.join(account_path, 'private_key.json'), 256)
        loaded = self.storage.load(self.acc.id)
        assert self.acc == loaded

    def test_update_regr(self):
        if False:
            while True:
                i = 10
        self.storage.update_regr(self.acc)
        account_path = os.path.join(self.config.accounts_dir, self.acc.id)
        assert os.path.exists(account_path)
        assert os.path.exists(os.path.join(account_path, 'regr.json'))
        assert not os.path.exists(os.path.join(account_path, 'meta.json'))
        assert not os.path.exists(os.path.join(account_path, 'private_key.json'))

    def test_update_meta(self):
        if False:
            while True:
                i = 10
        self.storage.update_meta(self.acc)
        account_path = os.path.join(self.config.accounts_dir, self.acc.id)
        assert os.path.exists(account_path)
        assert os.path.exists(os.path.join(account_path, 'meta.json'))
        assert not os.path.exists(os.path.join(account_path, 'regr.json'))
        assert not os.path.exists(os.path.join(account_path, 'private_key.json'))

    def test_find_all(self):
        if False:
            while True:
                i = 10
        self.storage.save(self.acc, self.mock_client)
        assert [self.acc] == self.storage.find_all()

    def test_find_all_none_empty_list(self):
        if False:
            print('Hello World!')
        assert [] == self.storage.find_all()

    def test_find_all_accounts_dir_absent(self):
        if False:
            while True:
                i = 10
        os.rmdir(self.config.accounts_dir)
        assert [] == self.storage.find_all()

    def test_find_all_load_skips(self):
        if False:
            print('Hello World!')
        self.storage._load_for_server_path = mock.MagicMock(side_effect=['x', errors.AccountStorageError, 'z'])
        with mock.patch('certbot._internal.account.os.listdir') as mock_listdir:
            mock_listdir.return_value = ['x', 'y', 'z']
            assert ['x', 'z'] == self.storage.find_all()

    def test_load_non_existent_raises_error(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(errors.AccountNotFound):
            self.storage.load('missing')

    def _set_server(self, server):
        if False:
            while True:
                i = 10
        self.config.server = server
        from certbot._internal.account import AccountFileStorage
        self.storage = AccountFileStorage(self.config)

    def test_find_all_neither_exists(self):
        if False:
            for i in range(10):
                print('nop')
        self._set_server('https://acme-staging-v02.api.letsencrypt.org/directory')
        assert [] == self.storage.find_all()
        assert [] == self.storage.find_all()
        assert not os.path.islink(self.config.accounts_dir)

    def test_find_all_find_before_save(self):
        if False:
            while True:
                i = 10
        self._set_server('https://acme-staging-v02.api.letsencrypt.org/directory')
        assert [] == self.storage.find_all()
        self.storage.save(self.acc, self.mock_client)
        assert [self.acc] == self.storage.find_all()
        assert [self.acc] == self.storage.find_all()
        assert not os.path.islink(self.config.accounts_dir)
        prev_server_path = 'https://acme-staging.api.letsencrypt.org/directory'
        assert not os.path.isdir(self.config.accounts_dir_for_server_path(prev_server_path))

    def test_find_all_save_before_find(self):
        if False:
            print('Hello World!')
        self._set_server('https://acme-staging-v02.api.letsencrypt.org/directory')
        self.storage.save(self.acc, self.mock_client)
        assert [self.acc] == self.storage.find_all()
        assert [self.acc] == self.storage.find_all()
        assert not os.path.islink(self.config.accounts_dir)
        assert os.path.isdir(self.config.accounts_dir)
        prev_server_path = 'https://acme-staging.api.letsencrypt.org/directory'
        assert not os.path.isdir(self.config.accounts_dir_for_server_path(prev_server_path))

    def test_find_all_server_downgrade(self):
        if False:
            for i in range(10):
                print('nop')
        self._set_server('https://acme-staging-v02.api.letsencrypt.org/directory')
        assert [] == self.storage.find_all()
        self.storage.save(self.acc, self.mock_client)
        assert [self.acc] == self.storage.find_all()
        self._set_server('https://acme-staging.api.letsencrypt.org/directory')
        assert [] == self.storage.find_all()

    def test_upgrade_version_staging(self):
        if False:
            while True:
                i = 10
        self._set_server('https://acme-staging.api.letsencrypt.org/directory')
        self.storage.save(self.acc, self.mock_client)
        self._set_server('https://acme-staging-v02.api.letsencrypt.org/directory')
        assert [self.acc] == self.storage.find_all()

    def test_upgrade_version_production(self):
        if False:
            for i in range(10):
                print('nop')
        self._set_server('https://acme-v01.api.letsencrypt.org/directory')
        self.storage.save(self.acc, self.mock_client)
        self._set_server('https://acme-v02.api.letsencrypt.org/directory')
        assert [self.acc] == self.storage.find_all()

    @mock.patch('certbot.compat.os.rmdir')
    def test_corrupted_account(self, mock_rmdir):
        if False:
            i = 10
            return i + 15
        self._set_server('https://acme-staging.api.letsencrypt.org/directory')
        self.storage.save(self.acc, self.mock_client)
        mock_rmdir.side_effect = OSError
        self.storage._load_for_server_path = mock.MagicMock(side_effect=errors.AccountStorageError)
        self._set_server('https://acme-staging-v02.api.letsencrypt.org/directory')
        assert [] == self.storage.find_all()

    def test_upgrade_load(self):
        if False:
            i = 10
            return i + 15
        self._set_server('https://acme-staging.api.letsencrypt.org/directory')
        self.storage.save(self.acc, self.mock_client)
        prev_account = self.storage.load(self.acc.id)
        self._set_server('https://acme-staging-v02.api.letsencrypt.org/directory')
        account = self.storage.load(self.acc.id)
        assert prev_account == account

    def test_upgrade_load_single_account(self):
        if False:
            print('Hello World!')
        self._set_server('https://acme-staging.api.letsencrypt.org/directory')
        self.storage.save(self.acc, self.mock_client)
        prev_account = self.storage.load(self.acc.id)
        self._set_server_and_stop_symlink('https://acme-staging-v02.api.letsencrypt.org/directory')
        account = self.storage.load(self.acc.id)
        assert prev_account == account

    def test_load_ioerror(self):
        if False:
            while True:
                i = 10
        self.storage.save(self.acc, self.mock_client)
        mock_open = mock.mock_open()
        mock_open.side_effect = IOError
        with mock.patch('builtins.open', mock_open):
            with pytest.raises(errors.AccountStorageError):
                self.storage.load(self.acc.id)

    def test_save_ioerrors(self):
        if False:
            return 10
        mock_open = mock.mock_open()
        mock_open.side_effect = IOError
        with mock.patch('builtins.open', mock_open):
            with pytest.raises(errors.AccountStorageError):
                self.storage.save(self.acc, self.mock_client)

    def test_delete(self):
        if False:
            while True:
                i = 10
        self.storage.save(self.acc, self.mock_client)
        self.storage.delete(self.acc.id)
        with pytest.raises(errors.AccountNotFound):
            self.storage.load(self.acc.id)

    def test_delete_no_account(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(errors.AccountNotFound):
            self.storage.delete(self.acc.id)

    def _assert_symlinked_account_removed(self):
        if False:
            return 10
        self._set_server('https://acme-staging.api.letsencrypt.org/directory')
        self.storage.save(self.acc, self.mock_client)
        with mock.patch('certbot._internal.constants.LE_REUSE_SERVERS', {}):
            self._set_server('https://acme-staging-v02.api.letsencrypt.org/directory')
            with pytest.raises(errors.AccountNotFound):
                self.storage.load(self.acc.id)

    def _test_delete_folders(self, server_url):
        if False:
            print('Hello World!')
        self._set_server('https://acme-staging.api.letsencrypt.org/directory')
        self.storage.save(self.acc, self.mock_client)
        self._set_server('https://acme-staging-v02.api.letsencrypt.org/directory')
        self.storage.load(self.acc.id)
        self._set_server(server_url)
        self.storage.delete(self.acc.id)
        self._set_server('https://acme-staging.api.letsencrypt.org/directory')
        with pytest.raises(errors.AccountNotFound):
            self.storage.load(self.acc.id)
        self._set_server('https://acme-staging-v02.api.letsencrypt.org/directory')
        with pytest.raises(errors.AccountNotFound):
            self.storage.load(self.acc.id)

    def test_delete_folders_up(self):
        if False:
            while True:
                i = 10
        self._test_delete_folders('https://acme-staging.api.letsencrypt.org/directory')
        self._assert_symlinked_account_removed()

    def test_delete_folders_down(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_delete_folders('https://acme-staging-v02.api.letsencrypt.org/directory')
        self._assert_symlinked_account_removed()

    def _set_server_and_stop_symlink(self, server_path):
        if False:
            print('Hello World!')
        self._set_server(server_path)
        with open(os.path.join(self.config.accounts_dir, 'foo'), 'w') as f:
            f.write('bar')

    def test_delete_shared_account_up(self):
        if False:
            return 10
        self._set_server_and_stop_symlink('https://acme-staging-v02.api.letsencrypt.org/directory')
        self._test_delete_folders('https://acme-staging.api.letsencrypt.org/directory')

    def test_delete_shared_account_down(self):
        if False:
            i = 10
            return i + 15
        self._set_server_and_stop_symlink('https://acme-staging-v02.api.letsencrypt.org/directory')
        self._test_delete_folders('https://acme-staging-v02.api.letsencrypt.org/directory')
if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:] + [__file__]))