import re
import pytest
import testutils
test_vars = testutils.securedrop_test_vars
testinfra_hosts = [test_vars.app_hostname, test_vars.monitor_hostname]

def test_fpf_apt_repo_present(host):
    if False:
        return 10
    '\n    Ensure the FPF apt repo, apt.freedom.press, is configured.\n    This repository is necessary for the SecureDrop Debian packages,\n    including:\n\n      * securedrop-app-code\n      * securedrop-keyring\n      * securedrop-grsec\n\n    Depending on the host, additional FPF-maintained packages will be\n    installed, e.g. for OSSEC. Install state for those packages\n    is tested separately.\n    '
    if test_vars.fpf_apt_repo_url == 'https://apt-test.freedom.press':
        f = host.file('/etc/apt/sources.list.d/apt_test_freedom_press.list')
    else:
        f = host.file('/etc/apt/sources.list.d/apt_freedom_press.list')
    repo_regex = '^deb \\[arch=amd64\\] {} {} main$'.format(re.escape(test_vars.fpf_apt_repo_url), re.escape(host.system_info.codename))
    assert f.contains(repo_regex)

def test_fpf_apt_repo_fingerprint(host):
    if False:
        for i in range(10):
            print('nop')
    "\n    Ensure the FPF apt repo has the correct fingerprint on the associated\n    signing pubkey. Recent key rotations have taken place in:\n\n      * 2016-10\n      * 2021-06\n\n    The old key has been removed, so only the new key's fingerprint should be\n    returned.\n    "
    c = host.run('apt-key finger')
    fpf_gpg_pub_key_info_old = '2224 5C81 E3BA EB41 38B3  6061 310F 5612 00F4 AD77'
    fpf_gpg_pub_key_info_new = '2359 E653 8C06 13E6 5295  5E6C 188E DD3B 7B22 E6A3'
    assert c.rc == 0
    assert fpf_gpg_pub_key_info_old not in c.stdout
    assert fpf_gpg_pub_key_info_new in c.stdout

@pytest.mark.parametrize('old_pubkey', ['pub   4096R/FC9F6818 2014-10-26 [expired: 2016-10-27]', 'pub   4096R/00F4AD77 2016-10-20 [expired: 2017-10-20]', 'pub   4096R/00F4AD77 2016-10-20 [expired: 2017-10-20]', 'pub   4096R/7B22E6A3 2021-05-10 [expired: 2022-07-04]', 'pub   4096R/7B22E6A3 2021-05-10 [expired: 2023-07-04]', 'uid                  Freedom of the Press Foundation Master Signing Key', 'B89A 29DB 2128 160B 8E4B  1B4C BADD E0C7 FC9F 6818'])
def test_fpf_apt_repo_old_pubkeys_absent(host, old_pubkey):
    if False:
        i = 10
        return i + 15
    '\n    Ensure that expired (or about-to-expire) public keys for the FPF\n    apt repo are NOT present. Updates to the securedrop-keyring package\n    should enforce clobbering of old pubkeys, and this check will confirm\n    absence.\n    '
    c = host.run('apt-key finger')
    assert old_pubkey not in c.stdout