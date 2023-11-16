import difflib
import os
import warnings
import pytest
import testutils
from jinja2 import Template
sdvars = testutils.securedrop_test_vars
testinfra_hosts = [sdvars.app_hostname, sdvars.monitor_hostname]

def test_ssh_motd_disabled(host):
    if False:
        while True:
            i = 10
    "\n    Ensure the SSH MOTD (Message of the Day) is disabled.\n    Grsecurity balks at Ubuntu's default MOTD.\n    "
    f = host.file('/etc/pam.d/sshd')
    assert f.is_file
    assert not f.contains('pam\\.motd')

def test_grsecurity_apt_packages(host):
    if False:
        i = 10
        return i + 15
    '\n    Ensure the grsecurity-related apt packages are present on the system.\n    Includes the FPF-maintained metapackage, as well as paxctl, for managing\n    PaX flags on binaries.\n    '
    assert host.package('securedrop-grsec').is_installed

@pytest.mark.parametrize('package', ['linux-signed-image-generic-lts-utopic', 'linux-signed-image-generic', 'linux-signed-generic-lts-utopic', 'linux-signed-generic', '^linux-image-.*generic$', '^linux-headers-.*'])
def test_generic_kernels_absent(host, package):
    if False:
        while True:
            i = 10
    '\n    Ensure the default Ubuntu-provided kernel packages are absent.\n    In the past, conflicting version numbers have caused machines\n    to reboot into a non-grsec kernel due to poor handling of\n    GRUB_DEFAULT logic. Removing the vendor-provided kernel packages\n    prevents accidental boots into non-grsec kernels.\n    '
    c = host.run(f'dpkg -l {package}')
    assert c.rc == 1
    error_text = f'dpkg-query: no packages found matching {package}'
    assert error_text in c.stderr.strip()

def test_grsecurity_lock_file(host):
    if False:
        while True:
            i = 10
    '\n    Ensure system is rerunning a grsecurity kernel by testing for the\n    `grsec_lock` file, which is automatically created by grsecurity.\n    '
    f = host.file('/proc/sys/kernel/grsecurity/grsec_lock')
    assert f.mode == 384
    assert f.user == 'root'
    assert f.size == 0

def test_grsecurity_kernel_is_running(host):
    if False:
        i = 10
        return i + 15
    '\n    Make sure the currently running kernel is our grsec kernel.\n    '
    c = host.run('uname -r')
    assert c.stdout.strip().endswith('-grsec-securedrop')

@pytest.mark.parametrize('sysctl_opt', [('kernel.grsecurity.grsec_lock', 1), ('kernel.grsecurity.rwxmap_logging', 0), ('vm.heap_stack_gap', 1048576)])
def test_grsecurity_sysctl_options(host, sysctl_opt):
    if False:
        i = 10
        return i + 15
    '\n    Check that the grsecurity-related sysctl options are set correctly.\n    In production the RWX logging is disabled, to reduce log noise.\n    '
    with host.sudo():
        assert host.sysctl(sysctl_opt[0]) == sysctl_opt[1]

def test_grsecurity_paxtest(host):
    if False:
        while True:
            i = 10
    '\n    Check that paxtest reports the expected mitigations. These are\n    "Killed" for most of the checks, with the notable exception of the\n    memcpy ones. Only newer versions of paxtest will fail the latter,\n    regardless of kernel.\n    '
    if not host.exists('/usr/bin/paxtest'):
        warnings.warn('Installing paxtest to run kernel tests')
        with host.sudo():
            host.run('apt-get update && apt-get install -y paxtest')
    try:
        with host.sudo():
            paxtest_cmd = 'paxtest blackhat /tmp/paxtest.log'
            paxtest_cmd += " | grep -P '^(Executable|Return)'"
            paxtest_results = host.check_output(paxtest_cmd)
        paxtest_template_path = '{}/paxtest_results.j2'.format(os.path.dirname(os.path.abspath(__file__)))
        memcpy_result = 'Killed'
        if host.system_info.codename == 'focal':
            memcpy_result = 'Vulnerable'
        with open(paxtest_template_path) as f:
            paxtest_template = Template(f.read().rstrip())
            paxtest_expected = paxtest_template.render(memcpy_result=memcpy_result)
        for paxtest_diff in difflib.context_diff(paxtest_expected.split('\n'), paxtest_results.split('\n')):
            print(paxtest_diff)
        assert paxtest_results == paxtest_expected
    finally:
        with host.sudo():
            host.run('apt-get remove -y paxtest')

def test_apt_autoremove(host):
    if False:
        return 10
    '\n    Ensure old packages have been autoremoved.\n    '
    c = host.run('apt-get --dry-run autoremove')
    assert c.rc == 0
    assert 'The following packages will be REMOVED' not in c.stdout

def test_paxctl(host):
    if False:
        return 10
    "\n    As of Focal, paxctl is not used, and shouldn't be installed.\n    "
    p = host.package('paxctl')
    assert not p.is_installed

def test_paxctld_focal(host):
    if False:
        i = 10
        return i + 15
    '\n    Focal-specific paxctld config checks.\n    Ensures paxctld is running and enabled, and relevant\n    exemptions are present in the config file.\n    '
    assert host.package('paxctld').is_installed
    f = host.file('/etc/paxctld.conf')
    assert f.is_file
    s = host.service('paxctld')
    assert s.is_enabled
    assert s.is_running
    assert host.file('/opt/securedrop/paxctld.conf').is_file
    hostname = host.check_output('hostname -s')
    assert 'app' in hostname or 'mon' in hostname
    if 'app' in hostname:
        assert f.contains('^/usr/sbin/apache2\tm')

@pytest.mark.parametrize('kernel_opts', ['WLAN', 'NFC', 'WIMAX', 'WIRELESS', 'HAMRADIO', 'IRDA', 'BT'])
def test_wireless_disabled_in_kernel_config(host, kernel_opts):
    if False:
        return 10
    "\n    Kernel modules for wireless are blacklisted, but we go one step further and\n    remove wireless support from the kernel. Let's make sure wireless is\n    disabled in the running kernel config!\n    "
    kernel_version = host.run('uname -r').stdout.strip()
    with host.sudo():
        kernel_config_path = f'/boot/config-{kernel_version}'
        kernel_config = host.file(kernel_config_path).content_string
        line = f'# CONFIG_{kernel_opts} is not set'
        assert line in kernel_config or kernel_opts not in kernel_config

@pytest.mark.parametrize('kernel_opts', ['CONFIG_X86_INTEL_TSX_MODE_OFF', 'CONFIG_PAX', 'CONFIG_GRKERNSEC'])
def test_kernel_options_enabled_config(host, kernel_opts):
    if False:
        while True:
            i = 10
    '\n    Tests kernel config for options that should be enabled\n    '
    kernel_version = host.run('uname -r').stdout.strip()
    with host.sudo():
        kernel_config_path = f'/boot/config-{kernel_version}'
        kernel_config = host.file(kernel_config_path).content_string
        line = f'{kernel_opts}=y'
        assert line in kernel_config

def test_mds_mitigations_and_smt_disabled(host):
    if False:
        return 10
    '\n    Ensure that full mitigations are in place for MDS\n    see https://www.kernel.org/doc/html/latest/admin-guide/hw-vuln/mds.html\n    '
    with host.sudo():
        grub_config_path = '/boot/grub/grub.cfg'
        grub_config = host.file(grub_config_path)
        assert grub_config.contains('mds=full,nosmt')

def test_kernel_boot_options(host):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure command-line options for currently booted kernel are set.\n    '
    with host.sudo():
        f = host.file('/proc/cmdline')
        boot_opts = f.content_string.split()
    assert 'noefi' in boot_opts
    if host.system_info.codename == 'focal':
        assert 'ipv6.disable=1' in boot_opts