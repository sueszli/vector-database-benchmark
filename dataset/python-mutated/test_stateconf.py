import io
import os
import os.path
import attr
import pytest
import salt.config
import salt.loader
from salt.exceptions import SaltRenderError
REQUISITES = ['require', 'require_in', 'use', 'use_in', 'watch', 'watch_in']

@attr.s
class Renderer:
    tmp_path = attr.ib()

    def __call__(self, content, sls='', saltenv='base', argline='-G yaml . jinja', **kws):
        if False:
            print('Hello World!')
        root_dir = self.tmp_path
        state_tree_dir = self.tmp_path / 'state_tree'
        cache_dir = self.tmp_path / 'cachedir'
        state_tree_dir.mkdir()
        cache_dir.mkdir()
        config = salt.config.minion_config(None)
        config['root_dir'] = str(root_dir)
        config['state_events'] = False
        config['id'] = 'match'
        config['file_client'] = 'local'
        config['file_roots'] = dict(base=[str(state_tree_dir)])
        config['cachedir'] = str(cache_dir)
        config['test'] = False
        _renderers = salt.loader.render(config, {'config.get': lambda a, b: False})
        return _renderers['stateconf'](io.StringIO(content), saltenv=saltenv, sls=sls, argline=argline, renderers=salt.loader.render(config, {}), **kws)

@pytest.fixture
def renderer(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    return Renderer(tmp_path)

def test_state_config(renderer):
    if False:
        for i in range(10):
            print('nop')
    result = renderer('\n.sls_params:\n  stateconf.set:\n    - name1: value1\n    - name2: value2\n\n.extra:\n  stateconf:\n    - set\n    - name: value\n\n# --- end of state config ---\n\ntest:\n  cmd.run:\n    - name: echo name1={{sls_params.name1}} name2={{sls_params.name2}} {{extra.name}}\n    - cwd: /\n', sls='test')
    assert len(result) == 3
    assert 'test::sls_params' in result and 'test' in result
    assert 'test::extra' in result
    assert result['test']['cmd.run'][0]['name'] == 'echo name1=value1 name2=value2 value'

def test_sls_dir(renderer):
    if False:
        print('Hello World!')
    result = renderer('\ntest:\n  cmd.run:\n    - name: echo sls_dir={{sls_dir}}\n    - cwd: /\n', sls='path.to.sls')
    assert result['test']['cmd.run'][0]['name'] == 'echo sls_dir=path{}to'.format(os.sep)

def test_states_declared_with_shorthand_no_args(renderer):
    if False:
        for i in range(10):
            print('nop')
    result = renderer('\ntest:\n  cmd.run:\n    - name: echo testing\n    - cwd: /\ntest1:\n  pkg.installed\ntest2:\n  user.present\n')
    assert len(result) == 3
    for args in (result['test1']['pkg.installed'], result['test2']['user.present']):
        assert isinstance(args, list)
        assert len(args) == 0
    assert result['test']['cmd.run'][0]['name'] == 'echo testing'

def test_adding_state_name_arg_for_dot_state_id(renderer):
    if False:
        print('Hello World!')
    result = renderer('\n.test:\n  pkg.installed:\n    - cwd: /\n.test2:\n  pkg.installed:\n    - name: vim\n', sls='test')
    assert result['test::test']['pkg.installed'][0]['name'] == 'test'
    assert result['test::test2']['pkg.installed'][0]['name'] == 'vim'

def test_state_prefix(renderer):
    if False:
        i = 10
        return i + 15
    result = renderer('\n.test:\n  cmd.run:\n    - name: echo renamed\n    - cwd: /\n\nstate_id:\n  cmd:\n    - run\n    - name: echo not renamed\n    - cwd: /\n', sls='test')
    assert len(result) == 2
    assert 'test::test' in result
    assert 'state_id' in result

@pytest.mark.parametrize('req', REQUISITES)
def test_dot_state_id_in_requisites(req, renderer):
    if False:
        i = 10
        return i + 15
    result = renderer('\n.test:\n  cmd.run:\n    - name: echo renamed\n    - cwd: /\n\nstate_id:\n  cmd.run:\n    - name: echo not renamed\n    - cwd: /\n    - {}:\n      - cmd: .test\n\n'.format(req), sls='test')
    assert len(result) == 2
    assert 'test::test' in result
    assert 'state_id' in result
    assert result['state_id']['cmd.run'][2][req][0]['cmd'] == 'test::test'

@pytest.mark.parametrize('req', REQUISITES)
def test_relative_include_with_requisites(req, renderer):
    if False:
        print('Hello World!')
    result = renderer('\ninclude:\n  - some.helper\n  - .utils\n\nstate_id:\n  cmd.run:\n    - name: echo test\n    - cwd: /\n    - {}:\n      - cmd: .utils::some_state\n'.format(req), sls='test.work')
    assert result['include'][1] == {'base': 'test.utils'}
    assert result['state_id']['cmd.run'][2][req][0]['cmd'] == 'test.utils::some_state'

def test_relative_include_and_extend(renderer):
    if False:
        return 10
    result = renderer('\ninclude:\n  - some.helper\n  - .utils\n\nextend:\n  .utils::some_state:\n    cmd.run:\n      - name: echo overridden\n    ', sls='test.work')
    assert 'test.utils::some_state' in result['extend']

@pytest.mark.parametrize('req', REQUISITES)
def test_multilevel_relative_include_with_requisites(req, renderer):
    if False:
        i = 10
        return i + 15
    result = renderer('\ninclude:\n  - .shared\n  - ..utils\n  - ...helper\n\nstate_id:\n  cmd.run:\n    - name: echo test\n    - cwd: /\n    - {}:\n      - cmd: ..utils::some_state\n'.format(req), sls='test.nested.work')
    assert result['include'][0] == {'base': 'test.nested.shared'}
    assert result['include'][1] == {'base': 'test.utils'}
    assert result['include'][2] == {'base': 'helper'}
    assert result['state_id']['cmd.run'][2][req][0]['cmd'] == 'test.utils::some_state'

def test_multilevel_relative_include_beyond_top_level(renderer):
    if False:
        while True:
            i = 10
    pytest.raises(SaltRenderError, renderer, '\ninclude:\n  - ...shared\n', sls='test.work')

def test_start_state_generation(renderer):
    if False:
        for i in range(10):
            print('nop')
    result = renderer('\nA:\n  cmd.run:\n    - name: echo hello\n    - cwd: /\nB:\n  cmd.run:\n    - name: echo world\n    - cwd: /\n', sls='test', argline='-so yaml . jinja')
    assert len(result) == 4
    assert result['test::start']['stateconf.set'][0]['require_in'][0]['cmd'] == 'A'

def test_goal_state_generation(renderer):
    if False:
        print('Hello World!')
    result = renderer('\n{% for sid in "ABCDE": %}\n{{sid}}:\n  cmd.run:\n    - name: echo this is {{sid}}\n    - cwd: /\n{% endfor %}\n\n', sls='test.goalstate', argline='yaml . jinja')
    assert len(result) == len('ABCDE') + 1
    reqs = result['test.goalstate::goal']['stateconf.set'][0]['require']
    assert {next(iter(i.values())) for i in reqs} == set('ABCDE')

def test_implicit_require_with_goal_state(renderer):
    if False:
        return 10
    result = renderer('\n{% for sid in "ABCDE": %}\n{{sid}}:\n  cmd.run:\n    - name: echo this is {{sid}}\n    - cwd: /\n{% endfor %}\n\nF:\n  cmd.run:\n    - name: echo this is F\n    - cwd: /\n    - require:\n      - cmd: A\n      - cmd: B\n\nG:\n  cmd.run:\n    - name: echo this is G\n    - cwd: /\n    - require:\n      - cmd: D\n      - cmd: F\n', sls='test', argline='-o yaml . jinja')
    sids = 'ABCDEFG'[::-1]
    for (i, sid) in enumerate(sids):
        if i < len(sids) - 1:
            assert result[sid]['cmd.run'][2]['require'][0]['cmd'] == sids[i + 1]
    F_args = result['F']['cmd.run']
    assert len(F_args) == 3
    F_req = F_args[2]['require']
    assert len(F_req) == 3
    assert F_req[1]['cmd'] == 'A'
    assert F_req[2]['cmd'] == 'B'
    G_args = result['G']['cmd.run']
    assert len(G_args) == 3
    G_req = G_args[2]['require']
    assert len(G_req) == 3
    assert G_req[1]['cmd'] == 'D'
    assert G_req[2]['cmd'] == 'F'
    goal_args = result['test::goal']['stateconf.set']
    assert len(goal_args) == 1
    assert [next(iter(i.values())) for i in goal_args[0]['require']] == list('ABCDEFG')

def test_slsdir(renderer):
    if False:
        print('Hello World!')
    result = renderer('\nformula/woot.sls:\n  cmd.run:\n    - name: echo {{ slspath }}\n    - cwd: /\n', sls='formula.woot', argline='yaml . jinja')
    r = result['formula/woot.sls']['cmd.run'][0]['name']
    assert r == 'echo formula/woot'