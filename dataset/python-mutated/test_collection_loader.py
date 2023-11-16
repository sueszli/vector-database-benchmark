from __future__ import annotations
import inspect
import os
import pkgutil
import pytest
import re
import sys
from importlib import import_module
from ansible.modules import ping as ping_module
from ansible.utils.collection_loader import AnsibleCollectionConfig, AnsibleCollectionRef
from ansible.utils.collection_loader._collection_finder import _AnsibleCollectionFinder, _AnsibleCollectionLoader, _AnsibleCollectionNSPkgLoader, _AnsibleCollectionPkgLoader, _AnsibleCollectionPkgLoaderBase, _AnsibleCollectionRootPkgLoader, _AnsibleNSTraversable, _AnsiblePathHookFinder, _get_collection_name_from_path, _get_collection_role_path, _get_collection_metadata, _iter_modules_impl
from ansible.utils.collection_loader._collection_config import _EventSource
from unittest.mock import MagicMock, NonCallableMagicMock, patch

@pytest.fixture(autouse=True, scope='function')
def teardown(*args, **kwargs):
    if False:
        while True:
            i = 10
    yield
    reset_collections_loader_state()

@pytest.mark.filterwarnings('ignore:find_module\\(\\) is deprecated and slated for removal in Python 3\\.12; use find_spec\\(\\) instead:DeprecationWarning', 'ignore:FileFinder\\.find_loader\\(\\) is deprecated and slated for removal in Python 3\\.12; use find_spec\\(\\) instead:DeprecationWarning')
@pytest.mark.skipif(sys.version_info >= (3, 12), reason='Testing Python 2 codepath (find_module) on Python 3, <= 3.11')
def test_find_module_py3_lt_312():
    if False:
        for i in range(10):
            print('nop')
    dir_to_a_file = os.path.dirname(ping_module.__file__)
    path_hook_finder = _AnsiblePathHookFinder(_AnsibleCollectionFinder(), dir_to_a_file)
    assert path_hook_finder.find_spec('missing') is None
    assert path_hook_finder.find_module('missing') is None

@pytest.mark.skipif(sys.version_info < (3, 12), reason='Testing Python 2 codepath (find_module) on Python >= 3.12')
def test_find_module_py3_gt_311():
    if False:
        print('Hello World!')
    dir_to_a_file = os.path.dirname(ping_module.__file__)
    path_hook_finder = _AnsiblePathHookFinder(_AnsibleCollectionFinder(), dir_to_a_file)
    assert path_hook_finder.find_spec('missing') is None

def test_finder_setup():
    if False:
        for i in range(10):
            print('nop')
    f = _AnsibleCollectionFinder(paths='/bogus/bogus')
    assert isinstance(f._n_collection_paths, list)
    with patch.object(sys, 'path', ['/bogus', default_test_collection_paths[1], '/morebogus', default_test_collection_paths[0]]):
        with patch('os.path.isdir', side_effect=lambda x: b'bogus' not in x):
            f = _AnsibleCollectionFinder(paths=['/explicit', '/other'])
            assert f._n_collection_paths == ['/explicit', '/other', default_test_collection_paths[1], default_test_collection_paths[0]]
    configured_paths = ['/bogus']
    playbook_paths = ['/playbookdir']
    with patch.object(sys, 'path', ['/bogus', '/playbookdir']) and patch('os.path.isdir', side_effect=lambda x: b'bogus' in x):
        f = _AnsibleCollectionFinder(paths=configured_paths)
        assert f._n_collection_paths == configured_paths
        f.set_playbook_paths(playbook_paths)
        assert f._n_collection_paths == extend_paths(playbook_paths, 'collections') + configured_paths
        f.set_playbook_paths(playbook_paths[0])
        assert f._n_collection_paths == extend_paths(playbook_paths, 'collections') + configured_paths

def test_finder_not_interested():
    if False:
        for i in range(10):
            print('nop')
    f = get_default_finder()
    assert f.find_module('nothanks') is None
    assert f.find_module('nothanks.sub', path=['/bogus/dir']) is None

def test_finder_ns():
    if False:
        i = 10
        return i + 15
    f = _AnsibleCollectionFinder(paths=['/bogus/bogus'])
    loader = f.find_module('ansible_collections')
    assert isinstance(loader, _AnsibleCollectionRootPkgLoader)
    loader = f.find_module('ansible_collections.ansible', path=['/bogus/bogus'])
    assert isinstance(loader, _AnsibleCollectionNSPkgLoader)
    f = get_default_finder()
    loader = f.find_module('ansible_collections')
    assert isinstance(loader, _AnsibleCollectionRootPkgLoader)
    with pytest.raises(ValueError):
        f.find_module('ansible_collections', path=['whatever'])
    with pytest.raises(ValueError):
        f.find_module('ansible_collections.whatever', path=None)
    paths = [os.path.join(p, 'ansible_collections/nonexistns') for p in default_test_collection_paths]
    loader = f.find_module('ansible_collections.nonexistns', paths)
    assert loader is None

def test_loader_remove():
    if False:
        for i in range(10):
            print('nop')
    fake_mp = [MagicMock(), _AnsibleCollectionFinder(), MagicMock(), _AnsibleCollectionFinder()]
    fake_ph = [MagicMock().m1, MagicMock().m2, _AnsibleCollectionFinder()._ansible_collection_path_hook, NonCallableMagicMock]
    with patch.object(sys, 'meta_path', fake_mp):
        with patch.object(sys, 'path_hooks', fake_ph):
            _AnsibleCollectionFinder()._remove()
            assert len(sys.meta_path) == 2
            assert all((not isinstance(mpf, _AnsibleCollectionFinder) for mpf in sys.meta_path))
            assert len(sys.path_hooks) == 3
            assert all((not isinstance(ph.__self__, _AnsibleCollectionFinder) for ph in sys.path_hooks if hasattr(ph, '__self__')))
            assert AnsibleCollectionConfig.collection_finder is None

def test_loader_install():
    if False:
        while True:
            i = 10
    fake_mp = [MagicMock(), _AnsibleCollectionFinder(), MagicMock(), _AnsibleCollectionFinder()]
    fake_ph = [MagicMock().m1, MagicMock().m2, _AnsibleCollectionFinder()._ansible_collection_path_hook, NonCallableMagicMock]
    with patch.object(sys, 'meta_path', fake_mp):
        with patch.object(sys, 'path_hooks', fake_ph):
            f = _AnsibleCollectionFinder()
            f._install()
            assert len(sys.meta_path) == 3
            assert sys.meta_path[0] is f
            assert all((not isinstance(mpf, _AnsibleCollectionFinder) for mpf in sys.meta_path[1:]))
            assert len(sys.path_hooks) == 4
            assert hasattr(sys.path_hooks[0], '__self__') and sys.path_hooks[0].__self__ is f
            assert all((not isinstance(ph.__self__, _AnsibleCollectionFinder) for ph in sys.path_hooks[1:] if hasattr(ph, '__self__')))
            assert AnsibleCollectionConfig.collection_finder is f
            with pytest.raises(ValueError):
                AnsibleCollectionConfig.collection_finder = f

def test_finder_coll():
    if False:
        for i in range(10):
            print('nop')
    f = get_default_finder()
    tests = [{'name': 'ansible_collections.testns.testcoll', 'test_paths': [default_test_collection_paths]}, {'name': 'ansible_collections.ansible.builtin', 'test_paths': [['/bogus'], default_test_collection_paths]}]
    for test_dict in tests:
        globals().update(test_dict)
        parent_pkg = name.rpartition('.')[0]
        for paths in test_paths:
            paths = [os.path.join(p, parent_pkg.replace('.', '/')) for p in paths]
            loader = f.find_module(name, path=paths)
            assert isinstance(loader, _AnsibleCollectionPkgLoader)

def test_root_loader_not_interested():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ImportError):
        _AnsibleCollectionRootPkgLoader('not_ansible_collections_toplevel', path_list=[])
    with pytest.raises(ImportError):
        _AnsibleCollectionRootPkgLoader('ansible_collections.somens', path_list=['/bogus'])

def test_root_loader():
    if False:
        i = 10
        return i + 15
    name = 'ansible_collections'
    for paths in ([], default_test_collection_paths):
        sys.modules.pop(name, None)
        loader = _AnsibleCollectionRootPkgLoader(name, paths)
        assert repr(loader).startswith('_AnsibleCollectionRootPkgLoader(path=')
        module = loader.load_module(name)
        assert module.__name__ == name
        assert module.__path__ == [p for p in extend_paths(paths, name) if os.path.isdir(p)]
        assert module.__file__ == '<ansible_synthetic_collection_package>'
        assert module.__package__ == name
        assert sys.modules.get(name) == module

def test_nspkg_loader_not_interested():
    if False:
        return 10
    with pytest.raises(ImportError):
        _AnsibleCollectionNSPkgLoader('not_ansible_collections_toplevel.something', path_list=[])
    with pytest.raises(ImportError):
        _AnsibleCollectionNSPkgLoader('ansible_collections.somens.somecoll', path_list=[])

def test_nspkg_loader_load_module():
    if False:
        print('Hello World!')
    for name in ['ansible_collections.ansible', 'ansible_collections.testns']:
        parent_pkg = name.partition('.')[0]
        module_to_load = name.rpartition('.')[2]
        paths = extend_paths(default_test_collection_paths, parent_pkg)
        existing_child_paths = [p for p in extend_paths(paths, module_to_load) if os.path.exists(p)]
        sys.modules.pop(name, None)
        loader = _AnsibleCollectionNSPkgLoader(name, path_list=paths)
        assert repr(loader).startswith('_AnsibleCollectionNSPkgLoader(path=')
        module = loader.load_module(name)
        assert module.__name__ == name
        assert isinstance(module.__loader__, _AnsibleCollectionNSPkgLoader)
        assert module.__path__ == existing_child_paths
        assert module.__package__ == name
        assert module.__file__ == '<ansible_synthetic_collection_package>'
        assert sys.modules.get(name) == module

def test_collpkg_loader_not_interested():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ImportError):
        _AnsibleCollectionPkgLoader('not_ansible_collections', path_list=[])
    with pytest.raises(ImportError):
        _AnsibleCollectionPkgLoader('ansible_collections.ns', path_list=['/bogus/bogus'])

def test_collpkg_loader_load_module():
    if False:
        return 10
    reset_collections_loader_state()
    with patch('ansible.utils.collection_loader.AnsibleCollectionConfig') as p:
        for name in ['ansible_collections.ansible.builtin', 'ansible_collections.testns.testcoll']:
            parent_pkg = name.rpartition('.')[0]
            module_to_load = name.rpartition('.')[2]
            paths = extend_paths(default_test_collection_paths, parent_pkg)
            existing_child_paths = [p for p in extend_paths(paths, module_to_load) if os.path.exists(p)]
            is_builtin = 'ansible.builtin' in name
            sys.modules.pop(name, None)
            loader = _AnsibleCollectionPkgLoader(name, path_list=paths)
            assert repr(loader).startswith('_AnsibleCollectionPkgLoader(path=')
            module = loader.load_module(name)
            assert module.__name__ == name
            assert isinstance(module.__loader__, _AnsibleCollectionPkgLoader)
            if is_builtin:
                assert module.__path__ == []
            else:
                assert module.__path__ == [existing_child_paths[0]]
            assert module.__package__ == name
            if is_builtin:
                assert module.__file__ == '<ansible_synthetic_collection_package>'
            else:
                assert module.__file__.endswith('__synthetic__') and os.path.isdir(os.path.dirname(module.__file__))
            assert sys.modules.get(name) == module
            assert hasattr(module, '_collection_meta') and isinstance(module._collection_meta, dict)
            assert module._collection_meta
            _collection_finder = import_module('ansible.utils.collection_loader._collection_finder')
            with patch.object(_collection_finder, '_meta_yml_to_dict', side_effect=Exception('bang')):
                with pytest.raises(Exception) as ex:
                    _AnsibleCollectionPkgLoader(name, path_list=paths).load_module(name)
                assert 'error parsing collection metadata' in str(ex.value)

def test_coll_loader():
    if False:
        print('Hello World!')
    with patch('ansible.utils.collection_loader.AnsibleCollectionConfig'):
        with pytest.raises(ValueError):
            _AnsibleCollectionLoader('ansible_collections')
        with pytest.raises(ValueError):
            _AnsibleCollectionLoader('ansible_collections.testns.testcoll', path_list=[])

def test_path_hook_setup():
    if False:
        return 10
    with patch.object(sys, 'path_hooks', []):
        found_hook = None
        pathhook_exc = None
        try:
            found_hook = _AnsiblePathHookFinder._get_filefinder_path_hook()
        except Exception as phe:
            pathhook_exc = phe
        assert str(pathhook_exc) == 'need exactly one FileFinder import hook (found 0)'
    assert repr(_AnsiblePathHookFinder(object(), '/bogus/path')) == "_AnsiblePathHookFinder(path='/bogus/path')"

def test_path_hook_importerror():
    if False:
        i = 10
        return i + 15
    reset_collections_loader_state()
    path_to_a_file = os.path.join(default_test_collection_paths[0], 'ansible_collections/testns/testcoll/plugins/action/my_action.py')
    assert _AnsiblePathHookFinder(_AnsibleCollectionFinder(), path_to_a_file).find_module('foo.bar.my_action') is None

def test_new_or_existing_module():
    if False:
        return 10
    module_name = 'blar.test.module'
    pkg_name = module_name.rpartition('.')[0]
    nuke_module_prefix(module_name)
    with _AnsibleCollectionPkgLoaderBase._new_or_existing_module(module_name, __package__=pkg_name) as new_module:
        assert sys.modules.get(module_name) is new_module
        assert new_module.__name__ == module_name
    assert sys.modules.get(module_name) is new_module
    with _AnsibleCollectionPkgLoaderBase._new_or_existing_module(module_name, __attr1__=42, blar='yo') as existing_module:
        assert sys.modules.get(module_name) is new_module
        assert hasattr(existing_module, '__package__') and existing_module.__package__ == pkg_name
        assert hasattr(existing_module, '__attr1__') and existing_module.__attr1__ == 42
        assert hasattr(existing_module, 'blar') and existing_module.blar == 'yo'
    with pytest.raises(ValueError) as ve:
        with _AnsibleCollectionPkgLoaderBase._new_or_existing_module(module_name) as existing_module:
            err_to_raise = ValueError('bang')
            raise err_to_raise
    assert ve.value is err_to_raise
    assert sys.modules.get(module_name) is existing_module
    nuke_module_prefix(module_name)
    with pytest.raises(ValueError) as ve:
        with _AnsibleCollectionPkgLoaderBase._new_or_existing_module(module_name) as new_module:
            err_to_raise = ValueError('bang')
            raise err_to_raise
    assert ve.value is err_to_raise
    assert sys.modules.get(module_name) is None

def test_iter_modules_impl():
    if False:
        return 10
    modules_trailer = 'ansible_collections/testns/testcoll/plugins'
    modules_pkg_prefix = modules_trailer.replace('/', '.') + '.'
    modules_path = os.path.join(default_test_collection_paths[0], modules_trailer)
    modules = list(_iter_modules_impl([modules_path], modules_pkg_prefix))
    assert modules
    assert set([('ansible_collections.testns.testcoll.plugins.action', True), ('ansible_collections.testns.testcoll.plugins.module_utils', True), ('ansible_collections.testns.testcoll.plugins.modules', True)]) == set(modules)
    modules_trailer = 'ansible_collections/testns/testcoll/plugins/modules'
    modules_pkg_prefix = modules_trailer.replace('/', '.') + '.'
    modules_path = os.path.join(default_test_collection_paths[0], modules_trailer)
    modules = list(_iter_modules_impl([modules_path], modules_pkg_prefix))
    assert modules
    assert len(modules) == 1
    assert modules[0][0] == 'ansible_collections.testns.testcoll.plugins.modules.amodule'
    assert modules[0][1] is False

def test_import_from_collection(monkeypatch):
    if False:
        return 10
    collection_root = os.path.join(os.path.dirname(__file__), 'fixtures', 'collections')
    collection_path = os.path.join(collection_root, 'ansible_collections/testns/testcoll/plugins/module_utils/my_util.py')
    expected_trace_log = [(collection_path, 5, 'call'), (collection_path, 6, 'line'), (collection_path, 6, 'return')]
    monkeypatch.setenv('ANSIBLE_COLLECTIONS_PATH', collection_root)
    finder = _AnsibleCollectionFinder(paths=[collection_root])
    reset_collections_loader_state(finder)
    from ansible_collections.testns.testcoll.plugins.module_utils.my_util import question
    original_trace_function = sys.gettrace()
    trace_log = []
    if original_trace_function:

        def my_trace_function(frame, event, arg):
            if False:
                return 10
            trace_log.append((frame.f_code.co_filename, frame.f_lineno, event))
            sys.settrace(original_trace_function)
            original_trace_function(frame, event, arg)
            sys.settrace(my_trace_function)
            return my_trace_function
    else:

        def my_trace_function(frame, event, arg):
            if False:
                while True:
                    i = 10
            trace_log.append((frame.f_code.co_filename, frame.f_lineno, event))
            return my_trace_function
    sys.settrace(my_trace_function)
    try:
        answer = question()
    finally:
        sys.settrace(original_trace_function)
    import ansible_collections.ansible.builtin.plugins.action as c2
    import ansible_collections.ansible.builtin.plugins as c3
    import ansible_collections.ansible.builtin as c4
    import ansible_collections.ansible as c5
    import ansible_collections as c6
    import ansible_collections.ansible.builtin.plugins.module_utils
    import ansible_collections.ansible.builtin.plugins.action
    assert ansible_collections.ansible.builtin.plugins.action == c3.action == c2
    import ansible_collections.ansible.builtin.plugins
    assert ansible_collections.ansible.builtin.plugins == c4.plugins == c3
    import ansible_collections.ansible.builtin
    assert ansible_collections.ansible.builtin == c5.builtin == c4
    import ansible_collections.ansible
    assert ansible_collections.ansible == c6.ansible == c5
    import ansible_collections
    assert ansible_collections == c6
    from ansible_collections.ansible import builtin
    from ansible_collections.ansible.builtin import plugins
    assert builtin.plugins == plugins
    from ansible_collections.ansible.builtin.plugins import action
    from ansible_collections.ansible.builtin.plugins.action import command
    assert action.command == command
    from ansible_collections.ansible.builtin.plugins.module_utils import basic
    from ansible_collections.ansible.builtin.plugins.module_utils.basic import AnsibleModule
    assert basic.AnsibleModule == AnsibleModule
    import ansible_collections.testns.testcoll.plugins.module_utils.my_other_util
    import ansible_collections.testns.testcoll.plugins.action.my_action
    assert inspect.get_annotations(question)['return'] is float
    assert trace_log == expected_trace_log

def test_eventsource():
    if False:
        while True:
            i = 10
    es = _EventSource()
    es.fire(42)
    handler1 = MagicMock()
    handler2 = MagicMock()
    es += handler1
    es.fire(99, my_kwarg='blah')
    handler1.assert_called_with(99, my_kwarg='blah')
    es += handler2
    es.fire(123, foo='bar')
    handler1.assert_called_with(123, foo='bar')
    handler2.assert_called_with(123, foo='bar')
    es -= handler2
    handler1.reset_mock()
    handler2.reset_mock()
    es.fire(123, foo='bar')
    handler1.assert_called_with(123, foo='bar')
    handler2.assert_not_called()
    es -= handler1
    handler1.reset_mock()
    es.fire('blah', kwarg=None)
    handler1.assert_not_called()
    handler2.assert_not_called()
    es -= handler1
    handler_bang = MagicMock(side_effect=Exception('bang'))
    es += handler_bang
    with pytest.raises(Exception) as ex:
        es.fire(123)
    assert 'bang' in str(ex.value)
    handler_bang.assert_called_with(123)
    with pytest.raises(ValueError):
        es += 42

def test_on_collection_load():
    if False:
        while True:
            i = 10
    finder = get_default_finder()
    reset_collections_loader_state(finder)
    load_handler = MagicMock()
    AnsibleCollectionConfig.on_collection_load += load_handler
    m = import_module('ansible_collections.testns.testcoll')
    load_handler.assert_called_once_with(collection_name='testns.testcoll', collection_path=os.path.dirname(m.__file__))
    _meta = _get_collection_metadata('testns.testcoll')
    assert _meta
    finder = get_default_finder()
    reset_collections_loader_state(finder)
    AnsibleCollectionConfig.on_collection_load += MagicMock(side_effect=Exception('bang'))
    with pytest.raises(Exception) as ex:
        import_module('ansible_collections.testns.testcoll')
    assert 'bang' in str(ex.value)

def test_default_collection_config():
    if False:
        print('Hello World!')
    finder = get_default_finder()
    reset_collections_loader_state(finder)
    assert AnsibleCollectionConfig.default_collection is None
    AnsibleCollectionConfig.default_collection = 'foo.bar'
    assert AnsibleCollectionConfig.default_collection == 'foo.bar'

def test_default_collection_detection():
    if False:
        return 10
    finder = get_default_finder()
    reset_collections_loader_state(finder)
    assert _get_collection_name_from_path('/') is None
    assert _get_collection_name_from_path('/foo/ansible_collections/bogusns/boguscoll/bar') is None
    live_collection_path = os.path.join(os.path.dirname(__file__), 'fixtures/collections/ansible_collections/testns/testcoll')
    assert _get_collection_name_from_path(live_collection_path) == 'testns.testcoll'
    live_collection_deep_path = os.path.join(live_collection_path, 'plugins/modules')
    assert _get_collection_name_from_path(live_collection_deep_path) == 'testns.testcoll'
    masked_collection_path = os.path.join(os.path.dirname(__file__), 'fixtures/collections_masked/ansible_collections/testns/testcoll')
    assert _get_collection_name_from_path(masked_collection_path) is None

@pytest.mark.parametrize('role_name,collection_list,expected_collection_name,expected_path_suffix', [('some_role', ['testns.testcoll', 'ansible.bogus'], 'testns.testcoll', 'testns/testcoll/roles/some_role'), ('testns.testcoll.some_role', ['ansible.bogus', 'testns.testcoll'], 'testns.testcoll', 'testns/testcoll/roles/some_role'), ('testns.testcoll.some_role', [], 'testns.testcoll', 'testns/testcoll/roles/some_role'), ('testns.testcoll.some_role', None, 'testns.testcoll', 'testns/testcoll/roles/some_role'), ('some_role', [], None, None), ('some_role', None, None, None)])
def test_collection_role_name_location(role_name, collection_list, expected_collection_name, expected_path_suffix):
    if False:
        print('Hello World!')
    finder = get_default_finder()
    reset_collections_loader_state(finder)
    expected_path = None
    if expected_path_suffix:
        expected_path = os.path.join(os.path.dirname(__file__), 'fixtures/collections/ansible_collections', expected_path_suffix)
    found = _get_collection_role_path(role_name, collection_list)
    if found:
        assert found[0] == role_name.rpartition('.')[2]
        assert found[1] == expected_path
        assert found[2] == expected_collection_name
    else:
        assert expected_collection_name is None and expected_path_suffix is None

def test_bogus_imports():
    if False:
        i = 10
        return i + 15
    finder = get_default_finder()
    reset_collections_loader_state(finder)
    bogus_imports = ['bogus_toplevel', 'ansible_collections.bogusns', 'ansible_collections.testns.boguscoll', 'ansible_collections.testns.testcoll.bogussub', 'ansible_collections.ansible.builtin.bogussub']
    for bogus_import in bogus_imports:
        with pytest.raises(ImportError):
            import_module(bogus_import)

def test_empty_vs_no_code():
    if False:
        while True:
            i = 10
    finder = get_default_finder()
    reset_collections_loader_state(finder)
    from ansible_collections.testns import testcoll
    from ansible_collections.testns.testcoll.plugins import module_utils
    assert testcoll.__loader__.get_source(testcoll.__name__) is None
    assert testcoll.__loader__.get_code(testcoll.__name__) is None
    assert module_utils.__loader__.get_source(module_utils.__name__) == b''
    assert module_utils.__loader__.get_code(module_utils.__name__) is not None

def test_finder_playbook_paths():
    if False:
        i = 10
        return i + 15
    finder = get_default_finder()
    reset_collections_loader_state(finder)
    import ansible_collections
    import ansible_collections.ansible
    import ansible_collections.testns
    assert hasattr(ansible_collections, '__path__') and len(ansible_collections.__path__) > 0
    assert hasattr(ansible_collections.ansible, '__path__') and len(ansible_collections.ansible.__path__) > 0
    assert hasattr(ansible_collections.testns, '__path__') and len(ansible_collections.testns.__path__) > 0
    with pytest.raises(ImportError):
        import ansible_collections.ansible.playbook_adj_other
    with pytest.raises(ImportError):
        import ansible_collections.testns.playbook_adj_other
    assert AnsibleCollectionConfig.playbook_paths == []
    playbook_path_fixture_dir = os.path.join(os.path.dirname(__file__), 'fixtures/playbook_path')
    AnsibleCollectionConfig.playbook_paths = [playbook_path_fixture_dir]
    assert AnsibleCollectionConfig.collection_paths[0] == os.path.join(playbook_path_fixture_dir, 'collections')
    assert ansible_collections.__path__[0] == os.path.join(playbook_path_fixture_dir, 'collections/ansible_collections')
    assert ansible_collections.ansible.__path__[0] == os.path.join(playbook_path_fixture_dir, 'collections/ansible_collections/ansible')
    assert all(('playbook_path' not in p for p in ansible_collections.testns.__path__))
    import ansible_collections.ansible.playbook_adj_other
    import ansible_collections.freshns.playbook_adj_other
    with pytest.raises(ImportError):
        import ansible_collections.testns.playbook_adj_other

def test_toplevel_iter_modules():
    if False:
        return 10
    finder = get_default_finder()
    reset_collections_loader_state(finder)
    modules = list(pkgutil.iter_modules(default_test_collection_paths, ''))
    assert len(modules) == 1
    assert modules[0][1] == 'ansible_collections'

def test_iter_modules_namespaces():
    if False:
        i = 10
        return i + 15
    finder = get_default_finder()
    reset_collections_loader_state(finder)
    paths = extend_paths(default_test_collection_paths, 'ansible_collections')
    modules = list(pkgutil.iter_modules(paths, 'ansible_collections.'))
    assert len(modules) == 2
    assert all((m[2] is True for m in modules))
    assert all((isinstance(m[0], _AnsiblePathHookFinder) for m in modules))
    assert set(['ansible_collections.testns', 'ansible_collections.ansible']) == set((m[1] for m in modules))

def test_collection_get_data():
    if False:
        return 10
    finder = get_default_finder()
    reset_collections_loader_state(finder)
    d = pkgutil.get_data('ansible_collections.testns.testcoll', 'plugins/action/my_action.py')
    assert b'hello from my_action.py' in d
    d = pkgutil.get_data('ansible_collections.testns.testcoll', 'bogus/bogus')
    assert d is None
    with pytest.raises(ValueError):
        plugins_pkg = import_module('ansible_collections.ansible.builtin')
        assert not os.path.exists(os.path.dirname(plugins_pkg.__file__))
        d = pkgutil.get_data('ansible_collections.ansible.builtin', 'plugins/connection/local.py')

@pytest.mark.parametrize('ref,ref_type,expected_collection,expected_subdirs,expected_resource,expected_python_pkg_name', [('ns.coll.myaction', 'action', 'ns.coll', '', 'myaction', 'ansible_collections.ns.coll.plugins.action'), ('ns.coll.subdir1.subdir2.myaction', 'action', 'ns.coll', 'subdir1.subdir2', 'myaction', 'ansible_collections.ns.coll.plugins.action.subdir1.subdir2'), ('ns.coll.myrole', 'role', 'ns.coll', '', 'myrole', 'ansible_collections.ns.coll.roles.myrole'), ('ns.coll.subdir1.subdir2.myrole', 'role', 'ns.coll', 'subdir1.subdir2', 'myrole', 'ansible_collections.ns.coll.roles.subdir1.subdir2.myrole')])
def test_fqcr_parsing_valid(ref, ref_type, expected_collection, expected_subdirs, expected_resource, expected_python_pkg_name):
    if False:
        print('Hello World!')
    assert AnsibleCollectionRef.is_valid_fqcr(ref, ref_type)
    r = AnsibleCollectionRef.from_fqcr(ref, ref_type)
    assert r.collection == expected_collection
    assert r.subdirs == expected_subdirs
    assert r.resource == expected_resource
    assert r.n_python_package_name == expected_python_pkg_name
    r = AnsibleCollectionRef.try_parse_fqcr(ref, ref_type)
    assert r.collection == expected_collection
    assert r.subdirs == expected_subdirs
    assert r.resource == expected_resource
    assert r.n_python_package_name == expected_python_pkg_name

@pytest.mark.parametrize(('fqcn', 'expected'), (('ns1.coll2', True), ('ns1#coll2', False), ('def.coll3', False), ('ns4.return', False), ('assert.this', False), ('import.that', False), ('.that', False), ('this.', False), ('.', False), ('', False)))
def test_fqcn_validation(fqcn, expected):
    if False:
        while True:
            i = 10
    'Vefiry that is_valid_collection_name validates FQCN correctly.'
    assert AnsibleCollectionRef.is_valid_collection_name(fqcn) is expected

@pytest.mark.parametrize('ref,ref_type,expected_error_type,expected_error_expression', [('no_dots_at_all_action', 'action', ValueError, 'is not a valid collection reference'), ('no_nscoll.myaction', 'action', ValueError, 'is not a valid collection reference'), ('no_nscoll%myaction', 'action', ValueError, 'is not a valid collection reference'), ('ns.coll.myaction', 'bogus', ValueError, 'invalid collection ref_type')])
def test_fqcr_parsing_invalid(ref, ref_type, expected_error_type, expected_error_expression):
    if False:
        for i in range(10):
            print('nop')
    assert not AnsibleCollectionRef.is_valid_fqcr(ref, ref_type)
    with pytest.raises(expected_error_type) as curerr:
        AnsibleCollectionRef.from_fqcr(ref, ref_type)
    assert re.search(expected_error_expression, str(curerr.value))
    r = AnsibleCollectionRef.try_parse_fqcr(ref, ref_type)
    assert r is None

@pytest.mark.parametrize('name,subdirs,resource,ref_type,python_pkg_name', [('ns.coll', None, 'res', 'doc_fragments', 'ansible_collections.ns.coll.plugins.doc_fragments'), ('ns.coll', 'subdir1', 'res', 'doc_fragments', 'ansible_collections.ns.coll.plugins.doc_fragments.subdir1'), ('ns.coll', 'subdir1.subdir2', 'res', 'action', 'ansible_collections.ns.coll.plugins.action.subdir1.subdir2')])
def test_collectionref_components_valid(name, subdirs, resource, ref_type, python_pkg_name):
    if False:
        i = 10
        return i + 15
    x = AnsibleCollectionRef(name, subdirs, resource, ref_type)
    assert x.collection == name
    if subdirs:
        assert x.subdirs == subdirs
    else:
        assert x.subdirs == ''
    assert x.resource == resource
    assert x.ref_type == ref_type
    assert x.n_python_package_name == python_pkg_name

@pytest.mark.parametrize('dirname,expected_result', [('become_plugins', 'become'), ('cache_plugins', 'cache'), ('connection_plugins', 'connection'), ('library', 'modules'), ('filter_plugins', 'filter'), ('bogus_plugins', ValueError), (None, ValueError)])
def test_legacy_plugin_dir_to_plugin_type(dirname, expected_result):
    if False:
        return 10
    if isinstance(expected_result, str):
        assert AnsibleCollectionRef.legacy_plugin_dir_to_plugin_type(dirname) == expected_result
    else:
        with pytest.raises(expected_result):
            AnsibleCollectionRef.legacy_plugin_dir_to_plugin_type(dirname)

@pytest.mark.parametrize('name,subdirs,resource,ref_type,expected_error_type,expected_error_expression', [('bad_ns', '', 'resource', 'action', ValueError, 'invalid collection name'), ('ns.coll.', '', 'resource', 'action', ValueError, 'invalid collection name'), ('ns.coll', 'badsubdir#', 'resource', 'action', ValueError, 'invalid subdirs entry'), ('ns.coll', 'badsubdir.', 'resource', 'action', ValueError, 'invalid subdirs entry'), ('ns.coll', '.badsubdir', 'resource', 'action', ValueError, 'invalid subdirs entry'), ('ns.coll', '', 'resource', 'bogus', ValueError, 'invalid collection ref_type')])
def test_collectionref_components_invalid(name, subdirs, resource, ref_type, expected_error_type, expected_error_expression):
    if False:
        while True:
            i = 10
    with pytest.raises(expected_error_type) as curerr:
        AnsibleCollectionRef(name, subdirs, resource, ref_type)
    assert re.search(expected_error_expression, str(curerr.value))

def test_importlib_resources():
    if False:
        while True:
            i = 10
    from importlib.resources import files
    from pathlib import Path
    f = get_default_finder()
    reset_collections_loader_state(f)
    ansible_collections_ns = files('ansible_collections')
    ansible_ns = files('ansible_collections.ansible')
    testns = files('ansible_collections.testns')
    testcoll = files('ansible_collections.testns.testcoll')
    testcoll2 = files('ansible_collections.testns.testcoll2')
    module_utils = files('ansible_collections.testns.testcoll.plugins.module_utils')
    assert isinstance(ansible_collections_ns, _AnsibleNSTraversable)
    assert isinstance(ansible_ns, _AnsibleNSTraversable)
    assert isinstance(testcoll, Path)
    assert isinstance(module_utils, Path)
    assert ansible_collections_ns.is_dir()
    assert ansible_ns.is_dir()
    assert testcoll.is_dir()
    assert module_utils.is_dir()
    first_path = Path(default_test_collection_paths[0])
    second_path = Path(default_test_collection_paths[1])
    testns_paths = []
    ansible_ns_paths = []
    for path in default_test_collection_paths[:2]:
        ansible_ns_paths.append(Path(path) / 'ansible_collections' / 'ansible')
        testns_paths.append(Path(path) / 'ansible_collections' / 'testns')
    assert testns._paths == testns_paths
    assert set(ansible_ns_paths).issubset(ansible_ns._paths)
    assert set((Path(p) / 'ansible_collections' for p in default_test_collection_paths[:2])).issubset(ansible_collections_ns._paths)
    assert testcoll2 == second_path / 'ansible_collections' / 'testns' / 'testcoll2'
    assert {p.name for p in module_utils.glob('*.py')} == {'__init__.py', 'my_other_util.py', 'my_util.py'}
    nestcoll_mu_init = first_path / 'ansible_collections' / 'testns' / 'testcoll' / 'plugins' / 'module_utils' / '__init__.py'
    assert next(module_utils.glob('__init__.py')) == nestcoll_mu_init
default_test_collection_paths = [os.path.join(os.path.dirname(__file__), 'fixtures', 'collections'), os.path.join(os.path.dirname(__file__), 'fixtures', 'collections_masked'), '/bogus/bogussub']

def get_default_finder():
    if False:
        i = 10
        return i + 15
    return _AnsibleCollectionFinder(paths=default_test_collection_paths)

def extend_paths(path_list, suffix):
    if False:
        for i in range(10):
            print('nop')
    suffix = suffix.replace('.', '/')
    return [os.path.join(p, suffix) for p in path_list]

def nuke_module_prefix(prefix):
    if False:
        print('Hello World!')
    for module_to_nuke in [m for m in sys.modules if m.startswith(prefix)]:
        sys.modules.pop(module_to_nuke)

def reset_collections_loader_state(metapath_finder=None):
    if False:
        for i in range(10):
            print('nop')
    _AnsibleCollectionFinder._remove()
    nuke_module_prefix('ansible_collections')
    nuke_module_prefix('ansible.modules')
    nuke_module_prefix('ansible.plugins')
    _AnsibleCollectionLoader._redirected_package_map = {}
    AnsibleCollectionConfig._default_collection = None
    AnsibleCollectionConfig._on_collection_load = _EventSource()
    if metapath_finder:
        metapath_finder._install()