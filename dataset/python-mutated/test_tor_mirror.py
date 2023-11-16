import pytest
import testutils
test_vars = testutils.securedrop_test_vars
testinfra_hosts = [test_vars.app_hostname, test_vars.monitor_hostname]

@pytest.mark.parametrize('repo_file', ['/etc/apt/sources.list.d/deb_torproject_org_torproject_org.list'])
def test_tor_mirror_absent(host, repo_file):
    if False:
        print('Hello World!')
    "\n    Ensure that neither the Tor Project repo, nor the FPF mirror of the\n    Tor Project repo, tor-apt.freedom.press, are configured. We've moved\n    to hosting Tor packages inside the primary FPF apt repo.\n    "
    f = host.file(repo_file)
    assert not f.exists

def test_tor_keyring_absent(host):
    if False:
        print('Hello World!')
    "\n    Tor packages are installed via the FPF apt mirror, and signed with the\n    SecureDrop Release Signing Key. As such, the official Tor public key\n    should *not* be present, since we don't want to install packages\n    from that source.\n    "
    package = 'deb.torproject.org-keyring'
    c = host.run(f'dpkg -l {package}')
    assert c.rc == 1
    error_text = f'dpkg-query: no packages found matching {package}'
    assert error_text in c.stderr.strip()

@pytest.mark.parametrize('tor_key_info', ['pub   2048R/886DDD89 2009-09-04 [expires: 2020-08-29]', 'Key fingerprint = A3C4 F0F9 79CA A22C DBA8  F512 EE8C BC9E 886D DD89', 'deb.torproject.org archive signing key'])
def test_tor_mirror_fingerprint(host, tor_key_info):
    if False:
        for i in range(10):
            print('nop')
    "\n    Legacy test. The Tor Project key was added to SecureDrop servers\n    via the `deb.torproject.org-keyring` package. Since FPF started mirroring\n    the official Tor apt repository, we no longer need the key to be present.\n\n    Since the `deb.torproject.org-keyring` package is installed on already\n    running instances, the public key will still be present. We'll need\n    to remove those packages separately.\n    "
    c = host.run('apt-key finger')
    assert c.rc == 0
    assert tor_key_info not in c.stdout

@pytest.mark.parametrize('repo_pattern', ['deb.torproject.org', 'tor-apt.freedom.press', 'tor-apt-test.freedom.press'])
def test_tor_repo_absent(host, repo_pattern):
    if False:
        while True:
            i = 10
    "\n    Ensure that no apt source list files contain the entry for\n    the official Tor apt repo, since we don't control issuing updates\n    in that repo. We're mirroring it to avoid breakage caused by\n    untested updates (which has broken prod twice to date).\n    "
    cmd = f"grep -rF '{repo_pattern}' /etc/apt/"
    c = host.run(cmd)
    assert c.rc != 0
    assert c.stdout == ''