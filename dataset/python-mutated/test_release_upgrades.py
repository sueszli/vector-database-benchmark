import testutils
test_vars = testutils.securedrop_test_vars
testinfra_hosts = [test_vars.app_hostname, test_vars.monitor_hostname]

def test_release_manager_installed(host):
    if False:
        for i in range(10):
            print('nop')
    '\n    The securedrop-config package munges `do-release-upgrade` settings\n    that assume the release-upgrader logic is installed. On hardware\n    installs of Ubuntu, it is, but the VM images we use in CI may\n    remove it to make the boxes leaner.\n    '
    assert host.package('ubuntu-release-upgrader-core').is_installed
    assert host.exists('do-release-upgrade')

def test_release_manager_upgrade_channel(host):
    if False:
        return 10
    '\n    Ensures that the `do-release-upgrade` command will not\n    suggest upgrades to a future LTS, until we test it and provide support.\n    '
    config_path = '/etc/update-manager/release-upgrades'
    assert host.file(config_path).is_file
    raw_output = host.check_output(f"grep '^Prompt' {config_path}")
    (_, channel) = raw_output.split('=')
    assert channel == 'never'