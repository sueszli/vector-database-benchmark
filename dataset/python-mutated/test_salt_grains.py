import packaging.version
import pytest
pytestmark = [pytest.mark.skip_on_windows]

def test_grains_items(salt_cli, salt_minion):
    if False:
        return 10
    '\n    Test grains.items\n    '
    ret = salt_cli.run('grains.items', minion_tgt=salt_minion.id)
    assert ret.data, ret
    assert 'osrelease' in ret.data

def test_grains_item_os(salt_cli, salt_minion):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test grains.item os\n    '
    ret = salt_cli.run('grains.item', 'os', minion_tgt=salt_minion.id)
    assert ret.data, ret
    assert 'os' in ret.data

def test_grains_item_pythonversion(salt_cli, salt_minion):
    if False:
        return 10
    '\n    Test grains.item pythonversion\n    '
    ret = salt_cli.run('grains.item', 'pythonversion', minion_tgt=salt_minion.id)
    assert ret.data, ret
    assert 'pythonversion' in ret.data

def test_grains_setval_key_val(salt_cli, salt_minion):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test grains.setval key val\n    '
    ret = salt_cli.run('grains.setval', 'key', 'val', minion_tgt=salt_minion.id)
    assert ret.data, ret
    assert 'key' in ret.data

def test_grains_package_onedir(salt_cli, salt_minion, install_salt):
    if False:
        while True:
            i = 10
    '\n    Test that the package grain returns onedir\n    '
    if packaging.version.parse(install_salt.version) < packaging.version.parse('3007.0'):
        pytest.skip("The package grain is only going to equal 'onedir' in version 3007.0 or later")
    ret = salt_cli.run('grains.get', 'package', minion_tgt=salt_minion.id)
    assert ret.data == 'onedir'
    assert ret.data, ret