import pytest
pytestmark = [pytest.mark.windows_whitelisted]

def test_pyobjects_renderer(state, state_tree, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test pyobjects renderer when running state.sls\n    '
    file_path = str(tmp_path).replace('\\', '/')
    sls1_contents = f'\n    #!pyobjects\n    import pathlib\n    import salt://test_pyobjects2.sls\n    test_file = pathlib.Path("{file_path}", "test")\n    File.managed(str(test_file))\n    '
    sls2_contents = f'\n    #!pyobjects\n    import pathlib\n    test_file = pathlib.Path("{file_path}", "test2")\n    File.managed(str(test_file))\n    '
    with pytest.helpers.temp_file('test_pyobjects.sls', sls1_contents, state_tree) as state1:
        with pytest.helpers.temp_file('test_pyobjects2.sls', sls2_contents, state_tree) as state2:
            ret = state.sls('test_pyobjects')
            assert not ret.errors
            for state_return in ret:
                assert state_return.result is True
                assert str(tmp_path) in state_return.name