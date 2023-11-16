"""
Pillar include tests
"""
import pytest

@pytest.fixture(scope='module')
def pillar_include_tree(base_env_pillar_tree_root_dir, salt_minion, salt_call_cli):
    if False:
        while True:
            i = 10
    top_file = "\n    base:\n      '{}':\n        - include\n        - glob-include\n        - include-c\n        - include-d\n    ".format(salt_minion.id)
    include_pillar_file = '\n    include:\n      - include-a:\n          key: element:a\n      - include-b:\n          key: element:b\n    '
    include_a_pillar_file = "\n    a:\n      - 'Entry A'\n    "
    include_b_pillar_file = "\n    b:\n      - 'Entry B'\n    "
    include_c_pillar_file = "\n    c:\n      - 'Entry C'\n    "
    include_d_pillar_file = '\n    include:\n      - include-c:\n          key: element:d\n    '
    top_tempfile = pytest.helpers.temp_file('top.sls', top_file, base_env_pillar_tree_root_dir)
    include_tempfile = pytest.helpers.temp_file('include.sls', include_pillar_file, base_env_pillar_tree_root_dir)
    include_a_tempfile = pytest.helpers.temp_file('include-a.sls', include_a_pillar_file, base_env_pillar_tree_root_dir)
    include_b_tempfile = pytest.helpers.temp_file('include-b.sls', include_b_pillar_file, base_env_pillar_tree_root_dir)
    include_c_tempfile = pytest.helpers.temp_file('include-c.sls', include_c_pillar_file, base_env_pillar_tree_root_dir)
    include_d_tempfile = pytest.helpers.temp_file('include-d.sls', include_d_pillar_file, base_env_pillar_tree_root_dir)
    glob_include_pillar_file = "\n    include:\n      - 'glob-include-*'\n    "
    glob_include_a_pillar_file = "\n    glob-a:\n      - 'Entry A'\n    "
    glob_include_b_pillar_file = "\n    glob-b:\n      - 'Entry B'\n    "
    top_tempfile = pytest.helpers.temp_file('top.sls', top_file, base_env_pillar_tree_root_dir)
    glob_include_tempfile = pytest.helpers.temp_file('glob-include.sls', glob_include_pillar_file, base_env_pillar_tree_root_dir)
    glob_include_a_tempfile = pytest.helpers.temp_file('glob-include-a.sls', glob_include_a_pillar_file, base_env_pillar_tree_root_dir)
    glob_include_b_tempfile = pytest.helpers.temp_file('glob-include-b.sls', glob_include_b_pillar_file, base_env_pillar_tree_root_dir)
    try:
        with top_tempfile, include_tempfile, include_a_tempfile, include_b_tempfile, include_c_tempfile, include_d_tempfile:
            with glob_include_tempfile, glob_include_a_tempfile, glob_include_b_tempfile:
                ret = salt_call_cli.run('saltutil.refresh_pillar', wait=True)
                assert ret.returncode == 0
                assert ret.data is True
                yield
    finally:
        ret = salt_call_cli.run('saltutil.refresh_pillar', wait=True)
        assert ret.returncode == 0
        assert ret.data is True

def test_pillar_include(pillar_include_tree, salt_call_cli):
    if False:
        return 10
    '\n    Test pillar include\n    '
    ret = salt_call_cli.run('pillar.items')
    assert ret.returncode == 0
    assert ret.data
    assert 'element' in ret.data
    assert 'a' in ret.data['element']
    assert ret.data['element']['a'] == {'a': ['Entry A']}
    assert 'b' in ret.data['element']
    assert ret.data['element']['b'] == {'b': ['Entry B']}

def test_pillar_glob_include(pillar_include_tree, salt_call_cli):
    if False:
        return 10
    '\n    Test pillar include via glob pattern\n    '
    ret = salt_call_cli.run('pillar.items')
    assert ret.returncode == 0
    assert ret.data
    assert 'glob-a' in ret.data
    assert ret.data['glob-a'] == ['Entry A']
    assert 'glob-b' in ret.data
    assert ret.data['glob-b'] == ['Entry B']

def test_pillar_include_already_included(pillar_include_tree, salt_call_cli):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test pillar include when a pillar file\n    has already been included.\n    '
    ret = salt_call_cli.run('pillar.items')
    assert ret.returncode == 0
    assert ret.data
    assert 'element' in ret.data
    assert 'd' in ret.data['element']
    assert ret.data['element']['d'] == {'c': ['Entry C']}