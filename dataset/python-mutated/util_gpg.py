import os
import pytest
import spack.util.gpg

@pytest.fixture()
def has_socket_dir():
    if False:
        i = 10
        return i + 15
    spack.util.gpg.init()
    return bool(spack.util.gpg.SOCKET_DIR)

def test_parse_gpg_output_case_one():
    if False:
        return 10
    output = 'sec::2048:1:AAAAAAAAAAAAAAAA:AAAAAAAAAA:AAAAAAAAAA:::::::::\nfpr:::::::::XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX:\nuid:::::::AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA::Joe (Test) <j.s@s.com>:\nssb::2048:1:AAAAAAAAAAAAAAAA:AAAAAAAAAA::::::::::\nsec::2048:1:AAAAAAAAAAAAAAAA:AAAAAAAAAA:AAAAAAAAAA:::::::::\nfpr:::::::::YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY:\nuid:::::::AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA::Joe (Test) <j.s@s.com>:\nssb::2048:1:AAAAAAAAAAAAAAAA:AAAAAAAAAA::::::::::\n'
    keys = spack.util.gpg._parse_secret_keys_output(output)
    assert len(keys) == 2
    assert keys[0] == 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    assert keys[1] == 'YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY'

def test_parse_gpg_output_case_two():
    if False:
        for i in range(10):
            print('nop')
    output = 'sec:-:2048:1:AAAAAAAAAA:AAAAAAAA:::-:::escaESCA:::+:::23::0:\nfpr:::::::::XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX:\ngrp:::::::::AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA:\nuid:-::::AAAAAAAAA::AAAAAAAAA::Joe (Test) <j.s@s.com>::::::::::0:\nssb:-:2048:1:AAAAAAAAA::::::esa:::+:::23:\nfpr:::::::::YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY:\ngrp:::::::::AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA:\n'
    keys = spack.util.gpg._parse_secret_keys_output(output)
    assert len(keys) == 1
    assert keys[0] == 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

def test_parse_gpg_output_case_three():
    if False:
        print('Hello World!')
    output = 'sec::2048:1:AAAAAAAAAAAAAAAA:AAAAAAAAAA:AAAAAAAAAA:::::::::\nfpr:::::::::WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW:\nuid:::::::AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA::Joe (Test) <j.s@s.com>:\nssb::2048:1:AAAAAAAAAAAAAAAA:AAAAAAAAAA::::::::::\nfpr:::::::::XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX:\nsec::2048:1:AAAAAAAAAAAAAAAA:AAAAAAAAAA:AAAAAAAAAA:::::::::\nfpr:::::::::YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY:\nuid:::::::AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA::Joe (Test) <j.s@s.com>:\nssb::2048:1:AAAAAAAAAAAAAAAA:AAAAAAAAAA::::::::::\nfpr:::::::::ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ:'
    keys = spack.util.gpg._parse_secret_keys_output(output)
    assert len(keys) == 2
    assert keys[0] == 'WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW'
    assert keys[1] == 'YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY'

@pytest.mark.requires_executables('gpg2')
def test_really_long_gnupghome_dir(tmpdir, has_socket_dir):
    if False:
        i = 10
        return i + 15
    if not has_socket_dir:
        pytest.skip('This test requires /var/run/user/$(id -u)')
    N = 960
    tdir = str(tmpdir)
    while len(tdir) < N:
        tdir = os.path.join(tdir, 'filler')
    tdir = tdir[:N].rstrip(os.sep)
    tdir += '0' * (N - len(tdir))
    with spack.util.gpg.gnupghome_override(tdir):
        spack.util.gpg.create(name='Spack testing 1', email='test@spack.io', comment='Spack testing key', expires='0')
        spack.util.gpg.list(True, True)