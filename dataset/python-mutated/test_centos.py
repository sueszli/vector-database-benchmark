import logging
import pytest
log = logging.getLogger(__name__)
pytestmark = [pytest.mark.destructive_test, pytest.mark.skip_if_not_root]
RPM_GPG_KEY_EPEL_8_SALTTEST = '-----BEGIN PGP PUBLIC KEY BLOCK-----\n\nmQINBFz3zvsBEADJOIIWllGudxnpvJnkxQz2CtoWI7godVnoclrdl83kVjqSQp+2\ndgxuG5mUiADUfYHaRQzxKw8efuQnwxzU9kZ70ngCxtmbQWGmUmfSThiapOz00018\n+eo5MFabd2vdiGo1y+51m2sRDpN8qdCaqXko65cyMuLXrojJHIuvRA/x7iqOrRfy\na8x3OxC4PEgl5pgDnP8pVK0lLYncDEQCN76D9ubhZQWhISF/zJI+e806V71hzfyL\n/Mt3mQm/li+lRKU25Usk9dWaf4NH/wZHMIPAkVJ4uD4H/uS49wqWnyiTYGT7hUbi\necF7crhLCmlRzvJR8mkRP6/4T/F3tNDPWZeDNEDVFUkTFHNU6/h2+O398MNY/fOh\nyKaNK3nnE0g6QJ1dOH31lXHARlpFOtWt3VmZU0JnWLeYdvap4Eff9qTWZJhI7Cq0\nWm8DgLUpXgNlkmquvE7P2W5EAr2E5AqKQoDbfw/GiWdRvHWKeNGMRLnGI3QuoX3U\npAlXD7v13VdZxNydvpeypbf/AfRyrHRKhkUj3cU1pYkM3DNZE77C5JUe6/0nxbt4\nETUZBTgLgYJGP8c7PbkVnO6I/KgL1jw+7MW6Az8Ox+RXZLyGMVmbW/TMc8haJfKL\nMoUo3TVk8nPiUhoOC0/kI7j9ilFrBxBU5dUtF4ITAWc8xnG6jJs/IsvRpQARAQAB\ntChGZWRvcmEgRVBFTCAoOCkgPGVwZWxAZmVkb3JhcHJvamVjdC5vcmc+iQI4BBMB\nAgAiBQJc9877AhsPBgsJCAcDAgYVCAIJCgsEFgIDAQIeAQIXgAAKCRAh6kWrL4bW\noWagD/4xnLWws34GByVDQkjprk0fX7Iyhpm/U7BsIHKspHLL+Y46vAAGY/9vMvdE\n0fcr9Ek2Zp7zE1RWmSCzzzUgTG6BFoTG1H4Fho/7Z8BXK/jybowXSZfqXnTOfhSF\nalwDdwlSJvfYNV9MbyvbxN8qZRU1z7PEWZrIzFDDToFRk0R71zHpnPTNIJ5/YXTw\nNqU9OxII8hMQj4ufF11040AJQZ7br3rzerlyBOB+Jd1zSPVrAPpeMyJppWFHSDAI\nWK6x+am13VIInXtqB/Cz4GBHLFK5d2/IYspVw47Solj8jiFEtnAq6+1Aq5WH3iB4\nbE2e6z00DSF93frwOyWN7WmPIoc2QsNRJhgfJC+isGQAwwq8xAbHEBeuyMG8GZjz\nxohg0H4bOSEujVLTjH1xbAG4DnhWO/1VXLX+LXELycO8ZQTcjj/4AQKuo4wvMPrv\n9A169oETG+VwQlNd74VBPGCvhnzwGXNbTK/KH1+WRH0YSb+41flB3NKhMSU6dGI0\nSGtIxDSHhVVNmx2/6XiT9U/znrZsG5Kw8nIbbFz+9MGUUWgJMsd1Zl9R8gz7V9fp\nn7L7y5LhJ8HOCMsY/Z7/7HUs+t/A1MI4g7Q5g5UuSZdgi0zxukiWuCkLeAiAP4y7\nzKK4OjJ644NDcWCHa36znwVmkz3ixL8Q0auR15Oqq2BjR/fyog==\n=84m8\n-----END PGP PUBLIC KEY BLOCK-----\n'
RPM_GPG_KEY_EPEL_7_SALTTEST = '-----BEGIN PGP PUBLIC KEY BLOCK-----\nVersion: GnuPG v1.4.11 (GNU/Linux)\n\nmQINBFKuaIQBEAC1UphXwMqCAarPUH/ZsOFslabeTVO2pDk5YnO96f+rgZB7xArB\nOSeQk7B90iqSJ85/c72OAn4OXYvT63gfCeXpJs5M7emXkPsNQWWSju99lW+AqSNm\njYWhmRlLRGl0OO7gIwj776dIXvcMNFlzSPj00N2xAqjMbjlnV2n2abAE5gq6VpqP\nvFXVyfrVa/ualogDVmf6h2t4Rdpifq8qTHsHFU3xpCz+T6/dGWKGQ42ZQfTaLnDM\njToAsmY0AyevkIbX6iZVtzGvanYpPcWW4X0RDPcpqfFNZk643xI4lsZ+Y2Er9Yu5\nS/8x0ly+tmmIokaE0wwbdUu740YTZjCesroYWiRg5zuQ2xfKxJoV5E+Eh+tYwGDJ\nn6HfWhRgnudRRwvuJ45ztYVtKulKw8QQpd2STWrcQQDJaRWmnMooX/PATTjCBExB\n9dkz38Druvk7IkHMtsIqlkAOQMdsX1d3Tov6BE2XDjIG0zFxLduJGbVwc/6rIc95\nT055j36Ez0HrjxdpTGOOHxRqMK5m9flFbaxxtDnS7w77WqzW7HjFrD0VeTx2vnjj\nGqchHEQpfDpFOzb8LTFhgYidyRNUflQY35WLOzLNV+pV3eQ3Jg11UFwelSNLqfQf\nuFRGc+zcwkNjHh5yPvm9odR1BIfqJ6sKGPGbtPNXo7ERMRypWyRz0zi0twARAQAB\ntChGZWRvcmEgRVBFTCAoNykgPGVwZWxAZmVkb3JhcHJvamVjdC5vcmc+iQI4BBMB\nAgAiBQJSrmiEAhsPBgsJCAcDAgYVCAIJCgsEFgIDAQIeAQIXgAAKCRBqL66iNSxk\n5cfGD/4spqpsTjtDM7qpytKLHKruZtvuWiqt5RfvT9ww9GUUFMZ4ZZGX4nUXg49q\nixDLayWR8ddG/s5kyOi3C0uX/6inzaYyRg+Bh70brqKUK14F1BrrPi29eaKfG+Gu\nMFtXdBG2a7OtPmw3yuKmq9Epv6B0mP6E5KSdvSRSqJWtGcA6wRS/wDzXJENHp5re\n9Ism3CYydpy0GLRA5wo4fPB5uLdUhLEUDvh2KK//fMjja3o0L+SNz8N0aDZyn5Ax\nCU9RB3EHcTecFgoy5umRj99BZrebR1NO+4gBrivIfdvD4fJNfNBHXwhSH9ACGCNv\nHnXVjHQF9iHWApKkRIeh8Fr2n5dtfJEF7SEX8GbX7FbsWo29kXMrVgNqHNyDnfAB\nVoPubgQdtJZJkVZAkaHrMu8AytwT62Q4eNqmJI1aWbZQNI5jWYqc6RKuCK6/F99q\nthFT9gJO17+yRuL6Uv2/vgzVR1RGdwVLKwlUjGPAjYflpCQwWMAASxiv9uPyYPHc\nErSrbRG0wjIfAR3vus1OSOx3xZHZpXFfmQTsDP7zVROLzV98R3JwFAxJ4/xqeON4\nvCPFU6OsT3lWQ8w7il5ohY95wmujfr6lk89kEzJdOTzcn7DBbUru33CQMGKZ3Evt\nRjsC7FDbL017qxS+ZVA/HGkyfiu4cpgV8VUnbql5eAZ+1Ll6Dw==\n=hdPa\n-----END PGP PUBLIC KEY BLOCK-----\n'

@pytest.fixture
def pkgrepo(states, grains):
    if False:
        for i in range(10):
            print('nop')
    if grains['os_family'] != 'RedHat':
        raise pytest.skip.Exception("Test only for CentOS platforms, not '{}' based distributions.".format(grains['os_family']), _use_item_location=True)
    return states.pkgrepo

@pytest.fixture
def centos_state_tree(grains, pkgrepo, state_tree):
    if False:
        while True:
            i = 10
    if grains['os'] not in ('CentOS', 'CentOS Stream'):
        pytest.skip("Test only applicable to CentOS, not '{}'.".format(grains['os']))
    managed_sls_contents = "\n    {% if grains['osmajorrelease'] == 8 %}\n    epel-salttest:\n      pkgrepo.managed:\n        - humanname: Extra Packages for Enterprise Linux 8 - $basearch (salttest)\n        - comments:\n          - '#baseurl=http://download.fedoraproject.org/pub/epel/8/$basearch'\n        - mirrorlist: https://mirrors.fedoraproject.org/metalink?repo=epel-8&arch=$basearch\n        - failovermethod: priority\n        - enabled: 1\n        - gpgcheck: 1\n        - gpgkey: file:///etc/pki/rpm-gpg/RPM-GPG-KEY-EPEL-8-salttest\n        - require:\n          - file: /etc/pki/rpm-gpg/RPM-GPG-KEY-EPEL-8-salttest\n\n    /etc/pki/rpm-gpg/RPM-GPG-KEY-EPEL-8-salttest:\n      file.managed:\n        - source: salt://pkgrepo/files/RPM-GPG-KEY-EPEL-8-salttest\n        - user: root\n        - group: root\n        - mode: 644\n    {% elif grains['osmajorrelease'] == 7 %}\n    epel-salttest:\n      pkgrepo.managed:\n        - humanname: Extra Packages for Enterprise Linux 7 - $basearch (salttest)\n        - comments:\n          - '#baseurl=http://download.fedoraproject.org/pub/epel/7/$basearch'\n        - mirrorlist: https://mirrors.fedoraproject.org/metalink?repo=epel-7&arch=$basearch\n        - failovermethod: priority\n        - enabled: 1\n        - gpgcheck: 1\n        - gpgkey: file:///etc/pki/rpm-gpg/RPM-GPG-KEY-EPEL-7-salttest\n        - require:\n          - file: /etc/pki/rpm-gpg/RPM-GPG-KEY-EPEL-7-salttest\n\n    /etc/pki/rpm-gpg/RPM-GPG-KEY-EPEL-7-salttest:\n      file.managed:\n        - source: salt://pkgrepo/files/RPM-GPG-KEY-EPEL-7-salttest\n        - user: root\n        - group: root\n        - mode: 644\n    {% endif %}\n    "
    absend_sls_contents = '\n    epel-salttest:\n      pkgrepo:\n        - absent\n    '
    centos_7_gpg_key = pytest.helpers.temp_file('pkgrepo/files/RPM-GPG-KEY-EPEL-7-salttest', RPM_GPG_KEY_EPEL_7_SALTTEST, state_tree)
    centos_8_gpg_key = pytest.helpers.temp_file('pkgrepo/files/RPM-GPG-KEY-EPEL-8-salttest', RPM_GPG_KEY_EPEL_8_SALTTEST, state_tree)
    managed_state_file = pytest.helpers.temp_file('pkgrepo/managed.sls', managed_sls_contents, state_tree)
    absent_state_file = pytest.helpers.temp_file('pkgrepo/absent.sls', absend_sls_contents, state_tree)
    try:
        with centos_7_gpg_key, centos_8_gpg_key, managed_state_file, absent_state_file:
            yield
    finally:
        pass

@pytest.mark.requires_salt_states('pkgrepo.managed', 'pkgrepo.absent')
def test_pkgrepo_managed_absent(grains, modules, subtests, centos_state_tree):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test adding/removing a repo\n    '
    add_repo_test_passed = False
    with subtests.test('Add Repo'):
        ret = modules.state.sls('pkgrepo.managed')
        assert ret.failed is False
        for state in ret:
            assert state.result is True
        add_repo_test_passed = True
    with subtests.test('Remove Repo'):
        if add_repo_test_passed is False:
            pytest.skip('Adding the repo failed. Skipping.')
        ret = modules.state.sls('pkgrepo.absent')
        assert ret.failed is False
        for state in ret:
            assert state.result is True

@pytest.fixture
def pkgrepo_with_comments_name(pkgrepo):
    if False:
        return 10
    pkgrepo_name = 'examplerepo'
    try:
        yield pkgrepo_name
    finally:
        try:
            pkgrepo.absent(pkgrepo_name)
        except Exception:
            pass

def test_pkgrepo_with_comments(pkgrepo, pkgrepo_with_comments_name, subtests):
    if False:
        while True:
            i = 10
    '\n    Test adding a repo with comments\n    '
    kwargs = {'name': pkgrepo_with_comments_name, 'baseurl': 'http://example.com/repo', 'enabled': False, 'comments': ['This is a comment']}
    with subtests.test('Add repo'):
        ret = pkgrepo.managed(**kwargs.copy())
        assert ret.result is True
    with subtests.test('Modify comments'):
        kwargs['comments'].append('This is another comment')
        ret = pkgrepo.managed(**kwargs.copy())
        assert ret.result is True
        assert ret.changes == {'comments': {'old': ['This is a comment'], 'new': ['This is a comment', 'This is another comment']}}
    with subtests.test('Repeat last call'):
        ret = pkgrepo.managed(**kwargs.copy())
        assert ret.result is True
        assert not ret.changes
        assert ret.comment == "Package repo '{}' already configured".format(pkgrepo_with_comments_name)

@pytest.fixture
def copr_pkgrepo_with_comments_name(pkgrepo, grains):
    if False:
        i = 10
        return i + 15
    if grains['osfinger'] in ('CentOS Linux-7', 'Amazon Linux-2') or grains['os'] == 'VMware Photon OS':
        pytest.skip('copr plugin not installed on {} CI'.format(grains['osfinger']))
    if grains['os'] in ('CentOS Stream', 'AlmaLinux') and grains['osmajorrelease'] == 9:
        pytest.skip('No repo for {} in test COPR yet'.format(grains['osfinger']))
    pkgrepo_name = 'hello-copr'
    try:
        yield pkgrepo_name
    finally:
        try:
            pkgrepo.absent(copr='mymindstorm/hello')
        except Exception:
            pass

def test_copr_pkgrepo_with_comments(pkgrepo, copr_pkgrepo_with_comments_name, subtests):
    if False:
        i = 10
        return i + 15
    '\n    Test adding a repo with comments\n    '
    kwargs = {'name': copr_pkgrepo_with_comments_name, 'copr': 'mymindstorm/hello', 'enabled': False, 'comments': ['This is a comment']}
    with subtests.test('Add repo'):
        ret = pkgrepo.managed(**kwargs.copy())
        assert ret.result is True
    with subtests.test('Modify comments'):
        kwargs['comments'].append('This is another comment')
        ret = pkgrepo.managed(**kwargs.copy())
        assert ret.result is True
        assert ret.changes == {'comments': {'old': ['This is a comment'], 'new': ['This is a comment', 'This is another comment']}}
    with subtests.test('Repeat last call'):
        ret = pkgrepo.managed(**kwargs.copy())
        assert ret.result is True
        assert not ret.changes
        assert ret.comment == "Package repo '{}' already configured".format(copr_pkgrepo_with_comments_name)