import os
import pytest
_MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modules')

@pytest.mark.parametrize('exclude_args,exclude_hooks', (pytest.param(True, False, id='args'), pytest.param(False, True, id='hooks'), pytest.param(True, True, id='args-and-hooks')))
def test_module_exclusion(exclude_args, exclude_hooks, pyi_builder):
    if False:
        print('Hello World!')
    pyi_args = ['--paths', os.path.join(_MODULES_DIR, 'pyi_module_exclusion', 'modules')]
    if exclude_args:
        pyi_args += ['--exclude', 'mymodule_feature2', '--exclude', 'mymodule_feature3']
    if exclude_hooks:
        pyi_args += ['--additional-hooks-dir', os.path.join(_MODULES_DIR, 'pyi_module_exclusion', 'hooks')]
    pyi_builder.test_source('\n        import mymodule_main\n\n        # Feature #1 module should be included, and thus available\n        assert mymodule_main.feature1_available == True\n\n        # Feature #2 module should be excluded, and thus unavailable\n        assert mymodule_main.feature2_available == False\n\n        # Feature #3 module should be excluded, and thus unavailable\n        assert mymodule_main.feature3_available == False\n        ', pyi_args=pyi_args, run_from_path=True)