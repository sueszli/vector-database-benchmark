import os
from binascii import unhexlify, b2a_base64, a2b_base64
import pytest
from ...constants import *
from ...crypto.key import AESOCBRepoKey, AESOCBKeyfileKey, CHPOKeyfileKey, Passphrase
from ...crypto.keymanager import RepoIdMismatch, NotABorgKeyFile
from ...helpers import EXIT_ERROR
from ...helpers import bin_to_hex
from ...helpers import msgpack
from ...repository import Repository
from .. import key
from . import RK_ENCRYPTION, KF_ENCRYPTION, cmd, _extract_repository_id, _set_repository_id, generate_archiver_tests
pytest_generate_tests = lambda metafunc: generate_archiver_tests(metafunc, kinds='local,remote,binary')

def test_change_passphrase(archivers, request):
    if False:
        i = 10
        return i + 15
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    os.environ['BORG_NEW_PASSPHRASE'] = 'newpassphrase'
    cmd(archiver, 'key', 'change-passphrase')
    os.environ['BORG_PASSPHRASE'] = 'newpassphrase'
    cmd(archiver, 'rlist')

def test_change_location_to_keyfile(archivers, request):
    if False:
        for i in range(10):
            print('nop')
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    log = cmd(archiver, 'rinfo')
    assert '(repokey' in log
    cmd(archiver, 'key', 'change-location', 'keyfile')
    log = cmd(archiver, 'rinfo')
    assert '(key file' in log

def test_change_location_to_b2keyfile(archivers, request):
    if False:
        i = 10
        return i + 15
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', '--encryption=repokey-blake2-aes-ocb')
    log = cmd(archiver, 'rinfo')
    assert '(repokey BLAKE2b' in log
    cmd(archiver, 'key', 'change-location', 'keyfile')
    log = cmd(archiver, 'rinfo')
    assert '(key file BLAKE2b' in log

def test_change_location_to_repokey(archivers, request):
    if False:
        i = 10
        return i + 15
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', KF_ENCRYPTION)
    log = cmd(archiver, 'rinfo')
    assert '(key file' in log
    cmd(archiver, 'key', 'change-location', 'repokey')
    log = cmd(archiver, 'rinfo')
    assert '(repokey' in log

def test_change_location_to_b2repokey(archivers, request):
    if False:
        i = 10
        return i + 15
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', '--encryption=keyfile-blake2-aes-ocb')
    log = cmd(archiver, 'rinfo')
    assert '(key file BLAKE2b' in log
    cmd(archiver, 'key', 'change-location', 'repokey')
    log = cmd(archiver, 'rinfo')
    assert '(repokey BLAKE2b' in log

def test_key_export_keyfile(archivers, request):
    if False:
        print('Hello World!')
    archiver = request.getfixturevalue(archivers)
    export_file = archiver.output_path + '/exported'
    cmd(archiver, 'rcreate', KF_ENCRYPTION)
    repo_id = _extract_repository_id(archiver.repository_path)
    cmd(archiver, 'key', 'export', export_file)
    with open(export_file) as fd:
        export_contents = fd.read()
    assert export_contents.startswith('BORG_KEY ' + bin_to_hex(repo_id) + '\n')
    key_file = archiver.keys_path + '/' + os.listdir(archiver.keys_path)[0]
    with open(key_file) as fd:
        key_contents = fd.read()
    assert key_contents == export_contents
    os.unlink(key_file)
    cmd(archiver, 'key', 'import', export_file)
    with open(key_file) as fd:
        key_contents2 = fd.read()
    assert key_contents2 == key_contents

def test_key_import_keyfile_with_borg_key_file(archivers, request, monkeypatch):
    if False:
        i = 10
        return i + 15
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', KF_ENCRYPTION)
    exported_key_file = os.path.join(archiver.output_path, 'exported')
    cmd(archiver, 'key', 'export', exported_key_file)
    key_file = os.path.join(archiver.keys_path, os.listdir(archiver.keys_path)[0])
    with open(key_file) as fd:
        key_contents = fd.read()
    os.unlink(key_file)
    imported_key_file = os.path.join(archiver.output_path, 'imported')
    monkeypatch.setenv('BORG_KEY_FILE', imported_key_file)
    cmd(archiver, 'key', 'import', exported_key_file)
    assert not os.path.isfile(key_file), '"borg key import" should respect BORG_KEY_FILE'
    with open(imported_key_file) as fd:
        imported_key_contents = fd.read()
    assert imported_key_contents == key_contents

def test_key_export_repokey(archivers, request):
    if False:
        return 10
    archiver = request.getfixturevalue(archivers)
    export_file = archiver.output_path + '/exported'
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    repo_id = _extract_repository_id(archiver.repository_path)
    cmd(archiver, 'key', 'export', export_file)
    with open(export_file) as fd:
        export_contents = fd.read()
    assert export_contents.startswith('BORG_KEY ' + bin_to_hex(repo_id) + '\n')
    with Repository(archiver.repository_path) as repository:
        repo_key = AESOCBRepoKey(repository)
        repo_key.load(None, Passphrase.env_passphrase())
    backup_key = AESOCBKeyfileKey(key.TestKey.MockRepository())
    backup_key.load(export_file, Passphrase.env_passphrase())
    assert repo_key.crypt_key == backup_key.crypt_key
    with Repository(archiver.repository_path) as repository:
        repository.save_key(b'')
    cmd(archiver, 'key', 'import', export_file)
    with Repository(archiver.repository_path) as repository:
        repo_key2 = AESOCBRepoKey(repository)
        repo_key2.load(None, Passphrase.env_passphrase())
    assert repo_key2.crypt_key == repo_key2.crypt_key

def test_key_export_qr(archivers, request):
    if False:
        return 10
    archiver = request.getfixturevalue(archivers)
    export_file = archiver.output_path + '/exported.html'
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    repo_id = _extract_repository_id(archiver.repository_path)
    cmd(archiver, 'key', 'export', '--qr-html', export_file)
    with open(export_file, encoding='utf-8') as fd:
        export_contents = fd.read()
    assert bin_to_hex(repo_id) in export_contents
    assert export_contents.startswith('<!doctype html>')
    assert export_contents.endswith('</html>\n')

def test_key_export_directory(archivers, request):
    if False:
        return 10
    archiver = request.getfixturevalue(archivers)
    export_directory = archiver.output_path + '/exported'
    os.mkdir(export_directory)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    cmd(archiver, 'key', 'export', export_directory, exit_code=EXIT_ERROR)

def test_key_export_qr_directory(archivers, request):
    if False:
        print('Hello World!')
    archiver = request.getfixturevalue(archivers)
    export_directory = archiver.output_path + '/exported'
    os.mkdir(export_directory)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    cmd(archiver, 'key', 'export', '--qr-html', export_directory, exit_code=EXIT_ERROR)

def test_key_import_errors(archivers, request):
    if False:
        return 10
    archiver = request.getfixturevalue(archivers)
    export_file = archiver.output_path + '/exported'
    cmd(archiver, 'rcreate', KF_ENCRYPTION)
    cmd(archiver, 'key', 'import', export_file, exit_code=EXIT_ERROR)
    with open(export_file, 'w') as fd:
        fd.write('something not a key\n')
    if archiver.FORK_DEFAULT:
        cmd(archiver, 'key', 'import', export_file, exit_code=2)
    else:
        with pytest.raises(NotABorgKeyFile):
            cmd(archiver, 'key', 'import', export_file)
    with open(export_file, 'w') as fd:
        fd.write('BORG_KEY a0a0a0\n')
    if archiver.FORK_DEFAULT:
        cmd(archiver, 'key', 'import', export_file, exit_code=2)
    else:
        with pytest.raises(RepoIdMismatch):
            cmd(archiver, 'key', 'import', export_file)

def test_key_export_paperkey(archivers, request):
    if False:
        i = 10
        return i + 15
    archiver = request.getfixturevalue(archivers)
    repo_id = 'e294423506da4e1ea76e8dcdf1a3919624ae3ae496fddf905610c351d3f09239'
    export_file = archiver.output_path + '/exported'
    cmd(archiver, 'rcreate', KF_ENCRYPTION)
    _set_repository_id(archiver.repository_path, unhexlify(repo_id))
    key_file = archiver.keys_path + '/' + os.listdir(archiver.keys_path)[0]
    with open(key_file, 'w') as fd:
        fd.write(CHPOKeyfileKey.FILE_ID + ' ' + repo_id + '\n')
        fd.write(b2a_base64(b'abcdefghijklmnopqrstu').decode())
    cmd(archiver, 'key', 'export', '--paper', export_file)
    with open(export_file) as fd:
        export_contents = fd.read()
    assert export_contents == 'To restore key use borg key import --paper /path/to/repo\n\nBORG PAPER KEY v1\nid: 2 / e29442 3506da 4e1ea7 / 25f62a 5a3d41 - 02\n 1: 616263 646566 676869 6a6b6c 6d6e6f 707172 - 6d\n 2: 737475 - 88\n'

def test_key_import_paperkey(archivers, request):
    if False:
        for i in range(10):
            print('nop')
    archiver = request.getfixturevalue(archivers)
    repo_id = 'e294423506da4e1ea76e8dcdf1a3919624ae3ae496fddf905610c351d3f09239'
    cmd(archiver, 'rcreate', KF_ENCRYPTION)
    _set_repository_id(archiver.repository_path, unhexlify(repo_id))
    key_file = archiver.keys_path + '/' + os.listdir(archiver.keys_path)[0]
    with open(key_file, 'w') as fd:
        fd.write(AESOCBKeyfileKey.FILE_ID + ' ' + repo_id + '\n')
        fd.write(b2a_base64(b'abcdefghijklmnopqrstu').decode())
    typed_input = b'2 / e29442 3506da 4e1ea7 / 25f62a 5a3d41  02\n2 / e29442 3506da 4e1ea7  25f62a 5a3d41 - 02\n2 / e29442 3506da 4e1ea7 / 25f62a 5a3d42 - 02\n2 / e29442 3506da 4e1ea7 / 25f62a 5a3d41 - 02\n616263 646566 676869 6a6b6c 6d6e6f 707172 - 6d\n\n\n737475 88\n73747i - 88\n73747 - 88\n73 74 75  -  89\n00a1 - 88\n2 / e29442 3506da 4e1ea7 / 25f62a 5a3d41 - 02\n616263 646566 676869 6a6b6c 6d6e6f 707172 - 6d\n73 74 75  -  88\n'
    cmd(archiver, 'key', 'import', '--paper', input=typed_input)
    typed_input = b'\ny\n'
    cmd(archiver, 'key', 'import', '--paper', input=typed_input)
    typed_input = b'2 / e29442 3506da 4e1ea7 / 25f62a 5a3d41 - 02\n\ny\n'
    cmd(archiver, 'key', 'import', '--paper', input=typed_input)

def test_init_defaults_to_argon2(archivers, request):
    if False:
        print('Hello World!')
    'https://github.com/borgbackup/borg/issues/747#issuecomment-1076160401'
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    with Repository(archiver.repository_path) as repository:
        key = msgpack.unpackb(a2b_base64(repository.load_key()))
        assert key['algorithm'] == 'argon2 chacha20-poly1305'

def test_change_passphrase_does_not_change_algorithm_argon2(archivers, request):
    if False:
        return 10
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    os.environ['BORG_NEW_PASSPHRASE'] = 'newpassphrase'
    cmd(archiver, 'key', 'change-passphrase')
    with Repository(archiver.repository_path) as repository:
        key = msgpack.unpackb(a2b_base64(repository.load_key()))
        assert key['algorithm'] == 'argon2 chacha20-poly1305'

def test_change_location_does_not_change_algorithm_argon2(archivers, request):
    if False:
        print('Hello World!')
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', KF_ENCRYPTION)
    cmd(archiver, 'key', 'change-location', 'repokey')
    with Repository(archiver.repository_path) as repository:
        key = msgpack.unpackb(a2b_base64(repository.load_key()))
        assert key['algorithm'] == 'argon2 chacha20-poly1305'