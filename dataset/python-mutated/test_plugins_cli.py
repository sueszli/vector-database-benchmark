import pytest
from httpie.status import ExitStatus
from tests.utils.plugins_cli import parse_listing

@pytest.mark.requires_installation
@pytest.mark.parametrize('cli_mode', [True, False])
def test_plugins_installation(httpie_plugins_success, interface, dummy_plugin, cli_mode):
    if False:
        while True:
            i = 10
    lines = httpie_plugins_success('install', dummy_plugin.path, cli_mode=cli_mode)
    assert lines[0].startswith(f'Installing {dummy_plugin.path}')
    assert f'Successfully installed {dummy_plugin.name}-{dummy_plugin.version}' in lines
    assert interface.is_installed(dummy_plugin.name)

@pytest.mark.requires_installation
def test_plugin_installation_with_custom_config(httpie_plugins_success, interface, dummy_plugin):
    if False:
        i = 10
        return i + 15
    interface.environment.config['default_options'] = ['--session-read-only', 'some-path.json', 'other', 'args']
    interface.environment.config.save()
    lines = httpie_plugins_success('install', dummy_plugin.path)
    assert lines[0].startswith(f'Installing {dummy_plugin.path}')
    assert f'Successfully installed {dummy_plugin.name}-{dummy_plugin.version}' in lines
    assert interface.is_installed(dummy_plugin.name)

@pytest.mark.requires_installation
@pytest.mark.parametrize('cli_mode', [True, False])
def test_plugins_listing(httpie_plugins_success, interface, dummy_plugin, cli_mode):
    if False:
        print('Hello World!')
    httpie_plugins_success('install', dummy_plugin.path, cli_mode=cli_mode)
    data = parse_listing(httpie_plugins_success('list'))
    assert data == {dummy_plugin.name: dummy_plugin.dump()}

@pytest.mark.requires_installation
def test_plugins_listing_multiple(interface, httpie_plugins_success, dummy_plugins):
    if False:
        return 10
    paths = [plugin.path for plugin in dummy_plugins]
    httpie_plugins_success('install', *paths)
    data = parse_listing(httpie_plugins_success('list'))
    assert data == {plugin.name: plugin.dump() for plugin in dummy_plugins}

@pytest.mark.requires_installation
@pytest.mark.parametrize('cli_mode', [True, False])
def test_plugins_uninstall(interface, httpie_plugins_success, dummy_plugin, cli_mode):
    if False:
        i = 10
        return i + 15
    httpie_plugins_success('install', dummy_plugin.path, cli_mode=cli_mode)
    httpie_plugins_success('uninstall', dummy_plugin.name, cli_mode=cli_mode)
    assert not interface.is_installed(dummy_plugin.name)

@pytest.mark.requires_installation
def test_plugins_listing_after_uninstall(interface, httpie_plugins_success, dummy_plugin):
    if False:
        for i in range(10):
            print('nop')
    httpie_plugins_success('install', dummy_plugin.path)
    httpie_plugins_success('uninstall', dummy_plugin.name)
    data = parse_listing(httpie_plugins_success('list'))
    assert len(data) == 0

@pytest.mark.requires_installation
def test_plugins_uninstall_specific(interface, httpie_plugins_success):
    if False:
        print('Hello World!')
    new_plugin_1 = interface.make_dummy_plugin()
    new_plugin_2 = interface.make_dummy_plugin()
    target_plugin = interface.make_dummy_plugin()
    httpie_plugins_success('install', new_plugin_1.path, new_plugin_2.path, target_plugin.path)
    httpie_plugins_success('uninstall', target_plugin.name)
    assert interface.is_installed(new_plugin_1.name)
    assert interface.is_installed(new_plugin_2.name)
    assert not interface.is_installed(target_plugin.name)

@pytest.mark.requires_installation
def test_plugins_installation_failed(httpie_plugins, interface):
    if False:
        i = 10
        return i + 15
    plugin = interface.make_dummy_plugin(build=False)
    result = httpie_plugins('install', plugin.path)
    assert result.exit_status == ExitStatus.ERROR
    assert result.stderr.splitlines()[-1].strip().startswith("Can't install")

@pytest.mark.requires_installation
def test_plugins_uninstall_non_existent(httpie_plugins, interface):
    if False:
        for i in range(10):
            print('nop')
    plugin = interface.make_dummy_plugin(build=False)
    result = httpie_plugins('uninstall', plugin.name)
    assert result.exit_status == ExitStatus.ERROR
    assert result.stderr.splitlines()[-1].strip() == f"Can't uninstall '{plugin.name}': package is not installed"

@pytest.mark.requires_installation
def test_plugins_double_uninstall(httpie_plugins, httpie_plugins_success, dummy_plugin):
    if False:
        return 10
    httpie_plugins_success('install', dummy_plugin.path)
    httpie_plugins_success('uninstall', dummy_plugin.name)
    result = httpie_plugins('uninstall', dummy_plugin.name)
    assert result.exit_status == ExitStatus.ERROR
    assert result.stderr.splitlines()[-1].strip() == f"Can't uninstall '{dummy_plugin.name}': package is not installed"

@pytest.mark.skip(reason='Doesn’t work in CI')
@pytest.mark.requires_installation
def test_plugins_upgrade(httpie_plugins, httpie_plugins_success, dummy_plugin):
    if False:
        for i in range(10):
            print('nop')
    httpie_plugins_success('install', dummy_plugin.path)
    dummy_plugin.version = '2.0.0'
    dummy_plugin.build()
    httpie_plugins_success('upgrade', dummy_plugin.path)
    data = parse_listing(httpie_plugins_success('list'))
    assert data[dummy_plugin.name]['version'] == '2.0.0'

@pytest.mark.requires_installation
def test_broken_plugins(httpie_plugins, httpie_plugins_success, dummy_plugin, broken_plugin):
    if False:
        i = 10
        return i + 15
    httpie_plugins_success('install', dummy_plugin.path, broken_plugin.path)
    with pytest.warns(UserWarning, match=f'While loading "{broken_plugin.name}", an error occurred: broken plugin'):
        data = parse_listing(httpie_plugins_success('list'))
        assert len(data) == 2
    with pytest.warns(UserWarning):
        httpie_plugins_success('uninstall', broken_plugin.name)
    data = parse_listing(httpie_plugins_success('list'))
    assert len(data) == 1