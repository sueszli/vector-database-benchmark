from __future__ import annotations
import configparser
import importlib.metadata
import sys
from unittest import mock
import pytest
from flake8.exceptions import ExecutionError
from flake8.exceptions import FailedToLoadPlugin
from flake8.plugins import finder
from flake8.plugins.pyflakes import FlakesChecker

def _ep(name='X', value='dne:dne', group='flake8.extension'):
    if False:
        return 10
    return importlib.metadata.EntryPoint(name, value, group)

def _plugin(package='local', version='local', ep=None):
    if False:
        while True:
            i = 10
    if ep is None:
        ep = _ep()
    return finder.Plugin(package, version, ep)

def _loaded(plugin=None, obj=None, parameters=None):
    if False:
        while True:
            i = 10
    if plugin is None:
        plugin = _plugin()
    if parameters is None:
        parameters = {'tree': True}
    return finder.LoadedPlugin(plugin, obj, parameters)

def test_loaded_plugin_entry_name_vs_display_name():
    if False:
        return 10
    loaded = _loaded(_plugin(package='package-name', ep=_ep(name='Q')))
    assert loaded.entry_name == 'Q'
    assert loaded.display_name == 'package-name[Q]'

def test_plugins_all_plugins():
    if False:
        print('Hello World!')
    tree_plugin = _loaded(parameters={'tree': True})
    logical_line_plugin = _loaded(parameters={'logical_line': True})
    physical_line_plugin = _loaded(parameters={'physical_line': True})
    report_plugin = _loaded(plugin=_plugin(ep=_ep(name='R', group='flake8.report')))
    plugins = finder.Plugins(checkers=finder.Checkers(tree=[tree_plugin], logical_line=[logical_line_plugin], physical_line=[physical_line_plugin]), reporters={'R': report_plugin}, disabled=[])
    assert tuple(plugins.all_plugins()) == (tree_plugin, logical_line_plugin, physical_line_plugin, report_plugin)

def test_plugins_versions_str():
    if False:
        for i in range(10):
            print('nop')
    plugins = finder.Plugins(checkers=finder.Checkers(tree=[_loaded(_plugin(package='pkg1', version='1'))], logical_line=[_loaded(_plugin(package='pkg2', version='2'))], physical_line=[_loaded(_plugin(package='pkg1', version='1'))]), reporters={'default': _loaded(_plugin(package='flake8')), 'custom': _loaded(_plugin(package='local'))}, disabled=[])
    assert plugins.versions_str() == 'pkg1: 1, pkg2: 2'

@pytest.fixture
def pyflakes_dist(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    metadata = 'Metadata-Version: 2.1\nName: pyflakes\nVersion: 9000.1.0\n'
    d = tmp_path.joinpath('pyflakes.dist-info')
    d.mkdir()
    d.joinpath('METADATA').write_text(metadata)
    return importlib.metadata.PathDistribution(d)

@pytest.fixture
def pycodestyle_dist(tmp_path):
    if False:
        print('Hello World!')
    metadata = 'Metadata-Version: 2.1\nName: pycodestyle\nVersion: 9000.2.0\n'
    d = tmp_path.joinpath('pycodestyle.dist-info')
    d.mkdir()
    d.joinpath('METADATA').write_text(metadata)
    return importlib.metadata.PathDistribution(d)

@pytest.fixture
def flake8_dist(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    metadata = 'Metadata-Version: 2.1\nName: flake8\nVersion: 9001\n'
    entry_points = '[console_scripts]\nflake8 = flake8.main.cli:main\n\n[flake8.extension]\nF = flake8.plugins.pyflakes:FlakesChecker\nE = flake8.plugins.pycodestyle:pycodestyle_logical\nW = flake8.plugins.pycodestyle:pycodestyle_physical\n\n[flake8.report]\ndefault = flake8.formatting.default:Default\npylint = flake8.formatting.default:Pylint\n'
    d = tmp_path.joinpath('flake8.dist-info')
    d.mkdir()
    d.joinpath('METADATA').write_text(metadata)
    d.joinpath('entry_points.txt').write_text(entry_points)
    return importlib.metadata.PathDistribution(d)

@pytest.fixture
def flake8_foo_dist(tmp_path):
    if False:
        print('Hello World!')
    metadata = 'Metadata-Version: 2.1\nName: flake8-foo\nVersion: 1.2.3\n'
    eps = '[console_scripts]\nfoo = flake8_foo:main\n[flake8.extension]\nQ = flake8_foo:Plugin\n[flake8.report]\nfoo = flake8_foo:Formatter\n'
    d = tmp_path.joinpath('flake8_foo.dist-info')
    d.mkdir()
    d.joinpath('METADATA').write_text(metadata)
    d.joinpath('entry_points.txt').write_text(eps)
    return importlib.metadata.PathDistribution(d)

@pytest.fixture
def mock_distribution(pyflakes_dist, pycodestyle_dist):
    if False:
        for i in range(10):
            print('nop')
    dists = {'pyflakes': pyflakes_dist, 'pycodestyle': pycodestyle_dist}
    with mock.patch.object(importlib.metadata, 'distribution', dists.get):
        yield

def test_flake8_plugins(flake8_dist, mock_distribution):
    if False:
        return 10
    'Ensure entrypoints for flake8 are parsed specially.'
    eps = flake8_dist.entry_points
    ret = set(finder._flake8_plugins(eps, 'flake8', '9001'))
    assert ret == {finder.Plugin('pyflakes', '9000.1.0', importlib.metadata.EntryPoint('F', 'flake8.plugins.pyflakes:FlakesChecker', 'flake8.extension')), finder.Plugin('pycodestyle', '9000.2.0', importlib.metadata.EntryPoint('E', 'flake8.plugins.pycodestyle:pycodestyle_logical', 'flake8.extension')), finder.Plugin('pycodestyle', '9000.2.0', importlib.metadata.EntryPoint('W', 'flake8.plugins.pycodestyle:pycodestyle_physical', 'flake8.extension')), finder.Plugin('flake8', '9001', importlib.metadata.EntryPoint('default', 'flake8.formatting.default:Default', 'flake8.report')), finder.Plugin('flake8', '9001', importlib.metadata.EntryPoint('pylint', 'flake8.formatting.default:Pylint', 'flake8.report'))}

def test_importlib_plugins(tmp_path, flake8_dist, flake8_foo_dist, mock_distribution, caplog):
    if False:
        i = 10
        return i + 15
    'Ensure we can load plugins from importlib.metadata.'
    flake8_colors_metadata = 'Metadata-Version: 2.1\nName: flake8-colors\nVersion: 1.2.3\n'
    flake8_colors_eps = '[flake8.extension]\nflake8-colors = flake8_colors:ColorFormatter\n'
    flake8_colors_d = tmp_path.joinpath('flake8_colors.dist-info')
    flake8_colors_d.mkdir()
    flake8_colors_d.joinpath('METADATA').write_text(flake8_colors_metadata)
    flake8_colors_d.joinpath('entry_points.txt').write_text(flake8_colors_eps)
    flake8_colors_dist = importlib.metadata.PathDistribution(flake8_colors_d)
    unrelated_metadata = 'Metadata-Version: 2.1\nName: unrelated\nVersion: 4.5.6\n'
    unrelated_eps = '[console_scripts]\nunrelated = unrelated:main\n'
    unrelated_d = tmp_path.joinpath('unrelated.dist-info')
    unrelated_d.mkdir()
    unrelated_d.joinpath('METADATA').write_text(unrelated_metadata)
    unrelated_d.joinpath('entry_points.txt').write_text(unrelated_eps)
    unrelated_dist = importlib.metadata.PathDistribution(unrelated_d)
    with mock.patch.object(importlib.metadata, 'distributions', return_value=[flake8_dist, flake8_colors_dist, flake8_foo_dist, unrelated_dist]):
        ret = set(finder._find_importlib_plugins())
    assert ret == {finder.Plugin('flake8-foo', '1.2.3', importlib.metadata.EntryPoint('Q', 'flake8_foo:Plugin', 'flake8.extension')), finder.Plugin('pycodestyle', '9000.2.0', importlib.metadata.EntryPoint('E', 'flake8.plugins.pycodestyle:pycodestyle_logical', 'flake8.extension')), finder.Plugin('pycodestyle', '9000.2.0', importlib.metadata.EntryPoint('W', 'flake8.plugins.pycodestyle:pycodestyle_physical', 'flake8.extension')), finder.Plugin('pyflakes', '9000.1.0', importlib.metadata.EntryPoint('F', 'flake8.plugins.pyflakes:FlakesChecker', 'flake8.extension')), finder.Plugin('flake8', '9001', importlib.metadata.EntryPoint('default', 'flake8.formatting.default:Default', 'flake8.report')), finder.Plugin('flake8', '9001', importlib.metadata.EntryPoint('pylint', 'flake8.formatting.default:Pylint', 'flake8.report')), finder.Plugin('flake8-foo', '1.2.3', importlib.metadata.EntryPoint('foo', 'flake8_foo:Formatter', 'flake8.report'))}
    assert caplog.record_tuples == [('flake8.plugins.finder', 30, 'flake8-colors plugin is obsolete in flake8>=5.0')]

def test_duplicate_dists(flake8_dist):
    if False:
        print('Hello World!')
    with mock.patch.object(importlib.metadata, 'distributions', return_value=[flake8_dist, flake8_dist]):
        ret = list(finder._find_importlib_plugins())
    assert len(ret) == len(set(ret))

def test_find_local_plugins_nothing():
    if False:
        while True:
            i = 10
    cfg = configparser.RawConfigParser()
    assert set(finder._find_local_plugins(cfg)) == set()

@pytest.fixture
def local_plugin_cfg():
    if False:
        while True:
            i = 10
    cfg = configparser.RawConfigParser()
    cfg.add_section('flake8:local-plugins')
    cfg.set('flake8:local-plugins', 'extension', 'Y=mod2:attr, X = mod:attr')
    cfg.set('flake8:local-plugins', 'report', 'Z=mod3:attr')
    return cfg

def test_find_local_plugins(local_plugin_cfg):
    if False:
        i = 10
        return i + 15
    ret = set(finder._find_local_plugins(local_plugin_cfg))
    assert ret == {finder.Plugin('local', 'local', importlib.metadata.EntryPoint('X', 'mod:attr', 'flake8.extension')), finder.Plugin('local', 'local', importlib.metadata.EntryPoint('Y', 'mod2:attr', 'flake8.extension')), finder.Plugin('local', 'local', importlib.metadata.EntryPoint('Z', 'mod3:attr', 'flake8.report'))}

def test_parse_plugin_options_not_specified(tmp_path):
    if False:
        i = 10
        return i + 15
    cfg = configparser.RawConfigParser()
    opts = finder.parse_plugin_options(cfg, str(tmp_path), enable_extensions=None, require_plugins=None)
    expected = finder.PluginOptions(local_plugin_paths=(), enable_extensions=frozenset(), require_plugins=frozenset())
    assert opts == expected

def test_parse_enabled_from_commandline(tmp_path):
    if False:
        print('Hello World!')
    cfg = configparser.RawConfigParser()
    cfg.add_section('flake8')
    cfg.set('flake8', 'enable_extensions', 'A,B,C')
    opts = finder.parse_plugin_options(cfg, str(tmp_path), enable_extensions='D,E,F', require_plugins=None)
    assert opts.enable_extensions == frozenset(('D', 'E', 'F'))

@pytest.mark.parametrize('opt', ('enable_extensions', 'enable-extensions'))
def test_parse_enabled_from_config(opt, tmp_path):
    if False:
        i = 10
        return i + 15
    cfg = configparser.RawConfigParser()
    cfg.add_section('flake8')
    cfg.set('flake8', opt, 'A,B,C')
    opts = finder.parse_plugin_options(cfg, str(tmp_path), enable_extensions=None, require_plugins=None)
    assert opts.enable_extensions == frozenset(('A', 'B', 'C'))

def test_parse_plugin_options_local_plugin_paths_missing(tmp_path):
    if False:
        while True:
            i = 10
    cfg = configparser.RawConfigParser()
    opts = finder.parse_plugin_options(cfg, str(tmp_path), enable_extensions=None, require_plugins=None)
    assert opts.local_plugin_paths == ()

def test_parse_plugin_options_local_plugin_paths(tmp_path):
    if False:
        print('Hello World!')
    cfg = configparser.RawConfigParser()
    cfg.add_section('flake8:local-plugins')
    cfg.set('flake8:local-plugins', 'paths', './a, ./b')
    opts = finder.parse_plugin_options(cfg, str(tmp_path), enable_extensions=None, require_plugins=None)
    expected = (str(tmp_path.joinpath('a')), str(tmp_path.joinpath('b')))
    assert opts.local_plugin_paths == expected

def test_find_plugins(tmp_path, flake8_dist, flake8_foo_dist, mock_distribution, local_plugin_cfg):
    if False:
        print('Hello World!')
    opts = finder.PluginOptions.blank()
    with mock.patch.object(importlib.metadata, 'distributions', return_value=[flake8_dist, flake8_foo_dist]):
        ret = finder.find_plugins(local_plugin_cfg, opts)
    assert ret == [finder.Plugin('flake8', '9001', importlib.metadata.EntryPoint('default', 'flake8.formatting.default:Default', 'flake8.report')), finder.Plugin('flake8', '9001', importlib.metadata.EntryPoint('pylint', 'flake8.formatting.default:Pylint', 'flake8.report')), finder.Plugin('flake8-foo', '1.2.3', importlib.metadata.EntryPoint('Q', 'flake8_foo:Plugin', 'flake8.extension')), finder.Plugin('flake8-foo', '1.2.3', importlib.metadata.EntryPoint('foo', 'flake8_foo:Formatter', 'flake8.report')), finder.Plugin('local', 'local', importlib.metadata.EntryPoint('X', 'mod:attr', 'flake8.extension')), finder.Plugin('local', 'local', importlib.metadata.EntryPoint('Y', 'mod2:attr', 'flake8.extension')), finder.Plugin('local', 'local', importlib.metadata.EntryPoint('Z', 'mod3:attr', 'flake8.report')), finder.Plugin('pycodestyle', '9000.2.0', importlib.metadata.EntryPoint('E', 'flake8.plugins.pycodestyle:pycodestyle_logical', 'flake8.extension')), finder.Plugin('pycodestyle', '9000.2.0', importlib.metadata.EntryPoint('W', 'flake8.plugins.pycodestyle:pycodestyle_physical', 'flake8.extension')), finder.Plugin('pyflakes', '9000.1.0', importlib.metadata.EntryPoint('F', 'flake8.plugins.pyflakes:FlakesChecker', 'flake8.extension'))]

def test_find_plugins_plugin_is_present(flake8_foo_dist):
    if False:
        print('Hello World!')
    cfg = configparser.RawConfigParser()
    options_flake8_foo_required = finder.PluginOptions(local_plugin_paths=(), enable_extensions=frozenset(), require_plugins=frozenset(('flake8-foo',)))
    options_not_required = finder.PluginOptions(local_plugin_paths=(), enable_extensions=frozenset(), require_plugins=frozenset())
    with mock.patch.object(importlib.metadata, 'distributions', return_value=[flake8_foo_dist]):
        finder.find_plugins(cfg, options_flake8_foo_required)
        finder.find_plugins(cfg, options_not_required)

def test_find_plugins_plugin_is_missing(flake8_dist, flake8_foo_dist):
    if False:
        return 10
    cfg = configparser.RawConfigParser()
    options_flake8_foo_required = finder.PluginOptions(local_plugin_paths=(), enable_extensions=frozenset(), require_plugins=frozenset(('flake8-foo',)))
    options_not_required = finder.PluginOptions(local_plugin_paths=(), enable_extensions=frozenset(), require_plugins=frozenset())
    with mock.patch.object(importlib.metadata, 'distributions', return_value=[flake8_dist]):
        finder.find_plugins(cfg, options_not_required)
        with pytest.raises(ExecutionError) as excinfo:
            finder.find_plugins(cfg, options_flake8_foo_required)
        (msg,) = excinfo.value.args
        assert msg == 'required plugins were not installed!\n- installed: flake8, pycodestyle, pyflakes\n- expected: flake8-foo\n- missing: flake8-foo'

def test_find_plugins_name_normalization(flake8_foo_dist):
    if False:
        i = 10
        return i + 15
    cfg = configparser.RawConfigParser()
    opts = finder.PluginOptions(local_plugin_paths=(), enable_extensions=frozenset(), require_plugins=frozenset(('Flake8_Foo',)))
    with mock.patch.object(importlib.metadata, 'distributions', return_value=[flake8_foo_dist]):
        finder.find_plugins(cfg, opts)

def test_parameters_for_class_plugin():
    if False:
        while True:
            i = 10
    'Verify that we can retrieve the parameters for a class plugin.'

    class FakeCheck:

        def __init__(self, tree):
            if False:
                while True:
                    i = 10
            raise NotImplementedError
    assert finder._parameters_for(FakeCheck) == {'tree': True}

def test_parameters_for_function_plugin():
    if False:
        print('Hello World!')
    'Verify that we retrieve the parameters for a function plugin.'

    def fake_plugin(physical_line, self, tree, optional=None):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError
    assert finder._parameters_for(fake_plugin) == {'physical_line': True, 'self': True, 'tree': True, 'optional': False}

def test_load_plugin_import_error():
    if False:
        print('Hello World!')
    plugin = _plugin(ep=_ep(value='dne:dne'))
    with pytest.raises(FailedToLoadPlugin) as excinfo:
        finder._load_plugin(plugin)
    (pkg, e) = excinfo.value.args
    assert pkg == 'local'
    assert isinstance(e, ModuleNotFoundError)

def test_load_plugin_not_callable():
    if False:
        return 10
    plugin = _plugin(ep=_ep(value='os:curdir'))
    with pytest.raises(FailedToLoadPlugin) as excinfo:
        finder._load_plugin(plugin)
    (pkg, e) = excinfo.value.args
    assert pkg == 'local'
    assert isinstance(e, TypeError)
    assert e.args == ('expected loaded plugin to be callable',)

def test_load_plugin_ok():
    if False:
        while True:
            i = 10
    plugin = _plugin(ep=_ep(value='flake8.plugins.pyflakes:FlakesChecker'))
    loaded = finder._load_plugin(plugin)
    assert loaded == finder.LoadedPlugin(plugin, FlakesChecker, {'tree': True, 'filename': True})

@pytest.fixture
def reset_sys():
    if False:
        return 10
    orig_path = sys.path[:]
    orig_modules = sys.modules.copy()
    yield
    sys.path[:] = orig_path
    sys.modules.clear()
    sys.modules.update(orig_modules)

@pytest.mark.usefixtures('reset_sys')
def test_import_plugins_extends_sys_path():
    if False:
        print('Hello World!')
    plugin = _plugin(ep=_ep(value='aplugin:ExtensionTestPlugin2'))
    opts = finder.PluginOptions(local_plugin_paths=('tests/integration/subdir',), enable_extensions=frozenset(), require_plugins=frozenset())
    ret = finder._import_plugins([plugin], opts)
    import aplugin
    assert ret == [finder.LoadedPlugin(plugin, aplugin.ExtensionTestPlugin2, {'tree': True})]

def test_classify_plugins():
    if False:
        return 10
    report_plugin = _loaded(plugin=_plugin(ep=_ep(name='R', group='flake8.report')))
    tree_plugin = _loaded(parameters={'tree': True})
    logical_line_plugin = _loaded(parameters={'logical_line': True})
    physical_line_plugin = _loaded(parameters={'physical_line': True})
    classified = finder._classify_plugins([report_plugin, tree_plugin, logical_line_plugin, physical_line_plugin], finder.PluginOptions.blank())
    assert classified == finder.Plugins(checkers=finder.Checkers(tree=[tree_plugin], logical_line=[logical_line_plugin], physical_line=[physical_line_plugin]), reporters={'R': report_plugin}, disabled=[])

def test_classify_plugins_enable_a_disabled_plugin():
    if False:
        while True:
            i = 10
    obj = mock.Mock(off_by_default=True)
    plugin = _plugin(ep=_ep(name='ABC'))
    loaded = _loaded(plugin=plugin, parameters={'tree': True}, obj=obj)
    normal_opts = finder.PluginOptions(local_plugin_paths=(), enable_extensions=frozenset(), require_plugins=frozenset())
    classified_normal = finder._classify_plugins([loaded], normal_opts)
    enabled_opts = finder.PluginOptions(local_plugin_paths=(), enable_extensions=frozenset(('ABC',)), require_plugins=frozenset())
    classified_enabled = finder._classify_plugins([loaded], enabled_opts)
    assert classified_normal == finder.Plugins(checkers=finder.Checkers([], [], []), reporters={}, disabled=[loaded])
    assert classified_enabled == finder.Plugins(checkers=finder.Checkers([loaded], [], []), reporters={}, disabled=[])

def test_classify_plugins_does_not_error_on_reporter_prefix():
    if False:
        while True:
            i = 10
    plugin = _plugin(ep=_ep(name='report-er', group='flake8.report'))
    loaded = _loaded(plugin=plugin)
    opts = finder.PluginOptions.blank()
    classified = finder._classify_plugins([loaded], opts)
    assert classified == finder.Plugins(checkers=finder.Checkers([], [], []), reporters={'report-er': loaded}, disabled=[])

def test_classify_plugins_errors_on_incorrect_checker_name():
    if False:
        return 10
    plugin = _plugin(ep=_ep(name='INVALID', group='flake8.extension'))
    loaded = _loaded(plugin=plugin, parameters={'tree': True})
    with pytest.raises(ExecutionError) as excinfo:
        finder._classify_plugins([loaded], finder.PluginOptions.blank())
    (msg,) = excinfo.value.args
    assert msg == 'plugin code for `local[INVALID]` does not match ^[A-Z]{1,3}[0-9]{0,3}$'

@pytest.mark.usefixtures('reset_sys')
def test_load_plugins():
    if False:
        print('Hello World!')
    plugin = _plugin(ep=_ep(value='aplugin:ExtensionTestPlugin2'))
    opts = finder.PluginOptions(local_plugin_paths=('tests/integration/subdir',), enable_extensions=frozenset(), require_plugins=frozenset())
    ret = finder.load_plugins([plugin], opts)
    import aplugin
    assert ret == finder.Plugins(checkers=finder.Checkers(tree=[finder.LoadedPlugin(plugin, aplugin.ExtensionTestPlugin2, {'tree': True})], logical_line=[], physical_line=[]), reporters={}, disabled=[])