import re
import pytest
import testutils
test_vars = testutils.securedrop_test_vars
testinfra_hosts = [test_vars.app_hostname, test_vars.monitor_hostname]
OFFSET_UPDATE = 2
OFFSET_UPGRADE = 1

def test_automatic_updates_dependencies(host):
    if False:
        print('Hello World!')
    '\n    Ensure critical packages are installed. If any of these are missing,\n    the system will fail to receive automatic updates.\n    In Focal, the apt config uses unattended-upgrades.\n    '
    assert host.package('unattended-upgrades').is_installed
    assert not host.package('cron-apt').is_installed
    assert not host.package('ntp').is_installed

def test_cron_apt_config(host):
    if False:
        print('Hello World!')
    '\n    Ensure custom cron-apt config is absent, as of Focal\n    '
    assert not host.file('/etc/cron-apt/config').exists
    assert not host.file('/etc/cron-apt/action.d/0-update').exists
    assert not host.file('/etc/cron-apt/action.d/5-security').exists
    assert not host.file('/etc/cron-apt/action.d/9-remove').exists
    assert not host.file('/etc/cron.d/cron-apt').exists
    assert not host.file('/etc/apt/security.list').exists
    assert not host.file('/etc/cron-apt/action.d/3-download').exists

@pytest.mark.parametrize('repo', ['deb http://security.ubuntu.com/ubuntu {securedrop_target_platform}-security main', 'deb http://security.ubuntu.com/ubuntu {securedrop_target_platform}-security universe', 'deb http://archive.ubuntu.com/ubuntu/ {securedrop_target_platform}-updates main', 'deb http://archive.ubuntu.com/ubuntu/ {securedrop_target_platform} main'])
def test_sources_list(host, repo):
    if False:
        print('Hello World!')
    '\n    Ensure the correct apt repositories are specified\n    in the sources.list for apt.\n    '
    repo_config = repo.format(securedrop_target_platform=host.system_info.codename)
    f = host.file('/etc/apt/sources.list')
    assert f.is_file
    assert f.user == 'root'
    assert f.mode == 420
    repo_regex = f'^{re.escape(repo_config)}$'
    assert f.contains(repo_regex)
apt_config_options = {'APT::Install-Recommends': 'false', 'Dpkg::Options': ['--force-confold', '--force-confdef'], 'APT::Periodic::Update-Package-Lists': '1', 'APT::Periodic::Unattended-Upgrade': '1', 'APT::Periodic::AutocleanInterval': '1', 'APT::Periodic::Enable': '1', 'Unattended-Upgrade::AutoFixInterruptedDpkg': 'true', 'Unattended-Upgrade::Automatic-Reboot': 'true', 'Unattended-Upgrade::Automatic-Reboot-Time': f'{test_vars.daily_reboot_time}:00', 'Unattended-Upgrade::Automatic-Reboot-WithUsers': 'true', 'Unattended-Upgrade::Origins-Pattern': ['origin=${distro_id},archive=${distro_codename}', 'origin=${distro_id},archive=${distro_codename}-security', 'origin=${distro_id},archive=${distro_codename}-updates', 'origin=SecureDrop,codename=${distro_codename}']}

@pytest.mark.parametrize(('k', 'v'), apt_config_options.items())
def test_unattended_upgrades_config(host, k, v):
    if False:
        return 10
    '\n    Ensures the apt and unattended-upgrades config is correct only under Ubuntu Focal\n    '
    c = host.run(f"apt-config dump --format '%v%n' {k}")
    assert c.rc == 0
    if hasattr(v, '__getitem__'):
        for i in v:
            assert i in c.stdout
    else:
        assert v in c.stdout

def test_unattended_securedrop_specific(host):
    if False:
        for i in range(10):
            print('nop')
    "\n    Ensures the 80securedrop config is correct. Under Ubuntu Focal,\n    it will include unattended-upgrade settings. Under all hosts,\n    it will disable installing 'recommended' packages.\n    "
    f = host.file('/etc/apt/apt.conf.d/80securedrop')
    assert f.is_file
    assert f.user == 'root'
    assert f.mode == 420
    assert f.contains('APT::Install-Recommends "false";')
    assert f.contains('Automatic-Reboot-Time')

def test_unattended_upgrades_functional(host):
    if False:
        print('Hello World!')
    '\n    Ensure unattended-upgrades completes successfully and ensures all packages\n    are up-to-date.\n    '
    c = host.run('sudo unattended-upgrades --dry-run --debug')
    assert c.rc == 0
    expected_origins = 'Allowed origins are: origin=Ubuntu,archive=focal, origin=Ubuntu,archive=focal-security, origin=Ubuntu,archive=focal-updates, origin=SecureDrop,codename=focal'
    expected_result = 'No packages found that can be upgraded unattended and no pending auto-removals'
    assert expected_origins in c.stdout
    assert expected_result in c.stdout

@pytest.mark.parametrize('service', ['apt-daily', 'apt-daily.timer', 'apt-daily-upgrade', 'apt-daily-upgrade.timer'])
def test_apt_daily_services_and_timers_enabled(host, service):
    if False:
        return 10
    '\n    Ensure the services and timers used for unattended upgrades are enabled\n    in Ubuntu 20.04 Focal.\n    '
    with host.sudo():
        s = host.service(service)
        assert s.is_enabled

def test_apt_daily_timer_schedule(host):
    if False:
        i = 10
        return i + 15
    "\n    Timer for running apt-daily, i.e. 'apt-get update', should be OFFSET_UPDATE hrs\n    before the daily_reboot_time.\n    "
    t = (int(test_vars.daily_reboot_time) - OFFSET_UPDATE) % 24
    c = host.run('systemctl show apt-daily.timer')
    assert 'TimersCalendar={ OnCalendar=*-*-* ' + f'{t:02d}' + ':00:00 ;' in c.stdout
    assert 'RandomizedDelayUSec=20m' in c.stdout

def test_apt_daily_upgrade_timer_schedule(host):
    if False:
        while True:
            i = 10
    "\n    Timer for running apt-daily-upgrade, i.e. 'apt-get upgrade', should be OFFSET_UPGRADE hrs\n    before the daily_reboot_time, and 1h after the apt-daily time.\n    "
    t = (int(test_vars.daily_reboot_time) - OFFSET_UPGRADE) % 24
    c = host.run('systemctl show apt-daily-upgrade.timer')
    assert 'TimersCalendar={ OnCalendar=*-*-* ' + f'{t:02d}' + ':00:00 ;' in c.stdout
    assert 'RandomizedDelayUSec=20m' in c.stdout

def test_reboot_required_cron(host):
    if False:
        for i in range(10):
            print('nop')
    "\n    Unattended-upgrades does not reboot the system if the updates don't require it.\n    However, we use daily reboots for SecureDrop to ensure memory is cleared periodically.\n    Here, we ensure that reboot-required flag is dropped twice daily to ensure the system\n    is rebooted every day at the scheduled time.\n    "
    f = host.file('/etc/cron.d/reboot-flag')
    assert f.is_file
    assert f.user == 'root'
    assert f.mode == 420
    line = '^{}$'.format(re.escape('0 */12 * * * root touch /var/run/reboot-required'))
    assert f.contains(line)

def test_all_packages_updated(host):
    if False:
        print('Hello World!')
    '\n    Ensure a safe-upgrade has already been run, by checking that no\n    packages are eligible for upgrade currently.\n\n    The Ansible config installs a specific, out-of-date version of Firefox\n    for use with Selenium. Therefore apt will report it\'s possible to upgrade\n    Firefox, which we\'ll need to mark as "OK" in terms of the tests.\n    '
    c = host.run('apt-get dist-upgrade --simulate')
    assert c.rc == 0
    assert '0 upgraded, 0 newly installed, 0 to remove' in c.stdout