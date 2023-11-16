import json
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, PropertyMock
import pytest
from pytest_mock import MockerFixture
from conda.gateways.connection import HTTPError
from conda.trust.signature_verification import SignatureError, _SignatureVerification
_TESTDATA = Path(__file__).parent / 'testdata'

@pytest.fixture
def initial_trust_root():
    if False:
        i = 10
        return i + 15
    return json.loads((_TESTDATA / '1.root.json').read_text())

def test_trusted_root_no_new_metadata(initial_trust_root: str, tmp_path: Path, mocker: MockerFixture):
    if False:
        return 10
    mocker.patch('conda.base.context.Context.av_data_dir', new_callable=PropertyMock, return_value=tmp_path)
    mocker.patch('conda.trust.signature_verification.INITIAL_TRUST_ROOT', new=initial_trust_root)
    sig_ver = _SignatureVerification()
    err = HTTPError()
    err.response = SimpleNamespace()
    err.response.status_code = 404
    sig_ver._fetch_channel_signing_data = MagicMock(side_effect=err)
    check_trusted_root = sig_ver.trusted_root
    sig_ver._fetch_channel_signing_data.assert_called()
    assert check_trusted_root == initial_trust_root

def test_trusted_root_2nd_metadata_on_disk_no_new_metadata_on_web(initial_trust_root: str, tmp_path: Path, mocker: MockerFixture):
    if False:
        i = 10
        return i + 15
    '\n    Tests a case where we cannot reach new root metadata online but have a newer version\n    locally (2.root.json).  As I understand it, we should use this new version if it is valid\n    '
    mocker.patch('conda.base.context.Context.av_data_dir', new_callable=PropertyMock, return_value=tmp_path)
    mocker.patch('conda.trust.signature_verification.INITIAL_TRUST_ROOT', new=initial_trust_root)
    sig_ver = _SignatureVerification()
    testdata_2_root = _TESTDATA / '2.root.json'
    test_2_root_dest = tmp_path / '2.root.json'
    copyfile(testdata_2_root, test_2_root_dest)
    err = HTTPError()
    err.response = SimpleNamespace()
    err.response.status_code = 404
    sig_ver._fetch_channel_signing_data = MagicMock(side_effect=err)
    check_trusted_root = sig_ver.trusted_root
    sig_ver._fetch_channel_signing_data.assert_called()
    test_2_root_data = json.loads(test_2_root_dest.read_text())
    assert check_trusted_root == test_2_root_data

def test_invalid_2nd_metadata_on_disk_no_new_metadata_on_web(initial_trust_root: str, tmp_path: Path, mocker: MockerFixture):
    if False:
        print('Hello World!')
    '\n    Unusual case:  We have an invalid 2.root.json on disk and no new metadata available online.  In this case,\n    our deliberate choice is to accept whatever on disk.\n    '
    mocker.patch('conda.base.context.Context.av_data_dir', new_callable=PropertyMock, return_value=tmp_path)
    mocker.patch('conda.trust.signature_verification.INITIAL_TRUST_ROOT', new=initial_trust_root)
    sig_ver = _SignatureVerification()
    testdata_2_root = _TESTDATA / '2.root_invalid.json'
    test_2_root_dest = tmp_path / '2.root.json'
    copyfile(testdata_2_root, test_2_root_dest)
    test_2_root_data = json.loads(test_2_root_dest.read_text())
    data_mock = Mock()
    data_mock.side_effect = [test_2_root_data]
    sig_ver = _SignatureVerification()
    sig_ver._fetch_channel_signing_data = data_mock
    check_trusted_root = sig_ver.trusted_root
    sig_ver._fetch_channel_signing_data.call_count == 1
    assert check_trusted_root == test_2_root_data

def test_2nd_root_metadata_from_web(initial_trust_root: str, tmp_path: Path, mocker: MockerFixture):
    if False:
        print('Hello World!')
    '\n    Test happy case where we get a new valid root metadata from the web\n    '
    mocker.patch('conda.base.context.Context.av_data_dir', new_callable=PropertyMock, return_value=tmp_path)
    mocker.patch('conda.trust.signature_verification.INITIAL_TRUST_ROOT', new=initial_trust_root)
    testdata_2_root = _TESTDATA / '2.root.json'
    test_2_root_data = json.loads(testdata_2_root.read_text())
    data_mock = Mock()
    data_mock.side_effect = [test_2_root_data]
    sig_ver = _SignatureVerification()
    sig_ver._fetch_channel_signing_data = data_mock
    check_trusted_root = sig_ver.trusted_root
    assert data_mock.call_count == 2
    assert check_trusted_root == test_2_root_data

def test_3rd_root_metadata_from_web(initial_trust_root: str, tmp_path: Path, mocker: MockerFixture):
    if False:
        i = 10
        return i + 15
    '\n    Test happy case where we get a chaing of valid root metadata from the web\n    '
    mocker.patch('conda.base.context.Context.av_data_dir', new_callable=PropertyMock, return_value=tmp_path)
    mocker.patch('conda.trust.signature_verification.INITIAL_TRUST_ROOT', new=initial_trust_root)
    testdata_2_root = _TESTDATA / '2.root.json'
    test_2_root_data = json.loads(testdata_2_root.read_text())
    testdata_3_root = _TESTDATA / '3.root.json'
    test_3_root_data = json.loads(testdata_3_root.read_text())
    data_mock = Mock()
    data_mock.side_effect = [test_2_root_data, test_3_root_data]
    sig_ver = _SignatureVerification()
    sig_ver._fetch_channel_signing_data = data_mock
    check_trusted_root = sig_ver.trusted_root
    assert data_mock.call_count == 3
    assert check_trusted_root == test_3_root_data

def test_single_invalid_signature_3rd_root_metadata_from_web(initial_trust_root: str, tmp_path: Path, mocker: MockerFixture):
    if False:
        print('Hello World!')
    '\n    Third root metadata retrieved from online has a bad signature. Test that we do not trust it.\n    '
    mocker.patch('conda.base.context.Context.av_data_dir', new_callable=PropertyMock, return_value=tmp_path)
    mocker.patch('conda.trust.signature_verification.INITIAL_TRUST_ROOT', new=initial_trust_root)
    testdata_2_root = _TESTDATA / '2.root.json'
    test_2_root_data = json.loads(testdata_2_root.read_text())
    testdata_3_root = _TESTDATA / '3.root_invalid.json'
    test_3_root_data = json.loads(testdata_3_root.read_text())
    data_mock = Mock()
    data_mock.side_effect = [test_2_root_data, test_3_root_data]
    sig_ver = _SignatureVerification()
    sig_ver._fetch_channel_signing_data = data_mock
    check_trusted_root = sig_ver.trusted_root
    assert data_mock.call_count == 2
    assert check_trusted_root == test_2_root_data

def test_trusted_root_no_new_key_mgr_online_key_mgr_is_on_disk(initial_trust_root: str, tmp_path: Path, mocker: MockerFixture):
    if False:
        while True:
            i = 10
    "\n    If we don't have a new key_mgr online, we use the one from disk\n    "
    mocker.patch('conda.base.context.Context.av_data_dir', new_callable=PropertyMock, return_value=tmp_path)
    mocker.patch('conda.trust.signature_verification.INITIAL_TRUST_ROOT', new=initial_trust_root)
    sig_ver = _SignatureVerification()
    err = HTTPError()
    err.response = SimpleNamespace()
    err.response.status_code = 404
    sig_ver._fetch_channel_signing_data = MagicMock(side_effect=err)
    test_key_mgr_path = _TESTDATA / 'key_mgr.json'
    test_key_mgr_dest = tmp_path / 'key_mgr.json'
    copyfile(test_key_mgr_path, test_key_mgr_dest)
    test_key_mgr_data = json.loads(test_key_mgr_path.read_text())
    check_key_mgr = sig_ver.key_mgr
    assert check_key_mgr == test_key_mgr_data

def test_trusted_root_no_new_key_mgr_online_key_mgr_not_on_disk(initial_trust_root: str, tmp_path: Path, mocker: MockerFixture):
    if False:
        while True:
            i = 10
    "\n    If we have no key_mgr online and no key_mgr on disk we don't have a key_mgr\n    "
    mocker.patch('conda.base.context.Context.av_data_dir', new_callable=PropertyMock, return_value=tmp_path)
    mocker.patch('conda.trust.signature_verification.INITIAL_TRUST_ROOT', new=initial_trust_root)
    sig_ver = _SignatureVerification()
    err = HTTPError()
    err.response = SimpleNamespace()
    err.response.status_code = 404
    sig_ver._fetch_channel_signing_data = MagicMock(side_effect=err)
    assert sig_ver.key_mgr is None

def test_trusted_root_new_key_mgr_online(initial_trust_root: str, tmp_path: Path, mocker: MockerFixture):
    if False:
        i = 10
        return i + 15
    '\n    We have a new key_mgr online that can be verified against our trusted root.\n    We should accept the new key_mgr\n    '
    mocker.patch('conda.base.context.Context.av_data_dir', new_callable=PropertyMock, return_value=tmp_path)
    mocker.patch('conda.trust.signature_verification.INITIAL_TRUST_ROOT', new=initial_trust_root)
    test_key_mgr_path = _TESTDATA / 'key_mgr.json'
    test_key_mgr_data = json.loads(test_key_mgr_path.read_text())
    err = HTTPError()
    err.response = SimpleNamespace()
    err.response.status_code = 404
    data_mock = Mock()
    data_mock.side_effect = [test_key_mgr_data, err]
    sig_ver = _SignatureVerification()
    if not sig_ver.enabled:
        pytest.skip('Signature verification not enabled')
    sig_ver._fetch_channel_signing_data = data_mock
    check_key_mgr = sig_ver.key_mgr
    assert check_key_mgr == test_key_mgr_data

def test_trusted_root_invalid_key_mgr_online_valid_on_disk(initial_trust_root: str, tmp_path: Path, mocker: MockerFixture):
    if False:
        return 10
    '\n    We have a new key_mgr online that can be verified against our trusted root.\n    We should accept the new key_mgr\n\n    Note:  This one does not fail with a warning and no side effects like the others.\n    Instead, we raise a SignatureError\n    '
    mocker.patch('conda.base.context.Context.av_data_dir', new_callable=PropertyMock, return_value=tmp_path)
    mocker.patch('conda.trust.signature_verification.INITIAL_TRUST_ROOT', new=initial_trust_root)
    sig_ver = _SignatureVerification()
    if not sig_ver.enabled:
        pytest.skip('Signature verification not enabled')
    test_key_mgr_invalid_path = _TESTDATA / 'key_mgr_invalid.json'
    test_key_mgr_invalid_data = json.loads(test_key_mgr_invalid_path.read_text())
    test_key_mgr_path = _TESTDATA / 'key_mgr.json'
    json.loads(test_key_mgr_path.read_text())
    test_key_mgr_dest = tmp_path / 'key_mgr.json'
    copyfile(test_key_mgr_path, test_key_mgr_dest)
    err = HTTPError()
    err.response = SimpleNamespace()
    err.response.status_code = 404
    data_mock = Mock()
    data_mock.side_effect = [test_key_mgr_invalid_data, err]
    sig_ver._fetch_channel_signing_data = data_mock
    with pytest.raises(SignatureError):
        pass