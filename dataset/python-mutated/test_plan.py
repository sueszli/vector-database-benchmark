from collections import defaultdict, namedtuple
from random import randint
import pytest
from pytest import MonkeyPatch
from pytest_mock import MockerFixture
import conda.instructions as inst
from conda import CondaError
from conda.base.context import conda_tests_ctxt_mgmt_def_pol, context
from conda.common.io import env_var
from conda.core.solve import get_pinned_specs
from conda.exceptions import PackagesNotFoundError
from conda.exports import execute_plan
from conda.models.channel import Channel
from conda.models.dist import Dist
from conda.models.match_spec import MatchSpec
from conda.models.records import PackageRecord
from conda.plan import _update_old_plan as update_old_plan
from conda.plan import add_unlink, display_actions
from conda.testing import CondaCLIFixture, TmpEnvFixture
from conda.testing.helpers import captured, get_index_r_1
(index, r) = get_index_r_1()
index = index.copy()

def get_matchspec_from_index(index, match_spec_str):
    if False:
        return 10
    ms = MatchSpec(match_spec_str)
    return next((prec for prec in index if ms.match(prec)))

def DPkg(s, **kwargs):
    if False:
        while True:
            i = 10
    d = Dist(s)
    _kwargs = dict(fn=d.to_filename(), name=d.name, version=d.version, build=d.build_string, build_number=int(d.build_string.rsplit('_', 1)[-1]), channel=d.channel, subdir=context.subdir, md5='012345789')
    _kwargs.update(kwargs)
    return PackageRecord(**_kwargs)

def solve(specs):
    if False:
        return 10
    return [Dist.from_string(fn) for fn in r.solve(specs)]

def test_simply_adds_unlink_on_non_windows():
    if False:
        print('Hello World!')
    actions = {}
    dist = Dist.from_string(f'foobar-{randint(100, 200)}-0')
    add_unlink(actions, dist)
    assert inst.UNLINK in actions
    assert actions[inst.UNLINK] == [dist]

def test_adds_to_existing_actions():
    if False:
        while True:
            i = 10
    actions = {inst.UNLINK: [{'foo': 'bar'}]}
    dist = Dist.from_string(f'foobar-{randint(100, 200)}-0')
    add_unlink(actions, dist)
    assert len(actions[inst.UNLINK]) == 2

def test_display_actions_0():
    if False:
        print('Hello World!')
    with env_var('CONDA_SHOW_CHANNEL_URLS', 'False', stack_callback=conda_tests_ctxt_mgmt_def_pol):
        actions = defaultdict(list)
        actions.update({'FETCH': [get_matchspec_from_index(index, 'channel-1::sympy==0.7.2=py27_0'), get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py27_0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be downloaded:\n\n    package                    |            build\n    ---------------------------|-----------------\n    sympy-0.7.2                |           py27_0         4.2 MB\n    numpy-1.7.1                |           py27_0         5.7 MB\n    ------------------------------------------------------------\n                                           Total:         9.9 MB\n\n'
        actions = defaultdict(list)
        actions.update({'PREFIX': '/Users/aaronmeurer/anaconda/envs/test', 'SYMLINK_CONDA': ['/Users/aaronmeurer/anaconda'], 'LINK': [get_matchspec_from_index(index, 'channel-1::python==3.3.2=0'), get_matchspec_from_index(index, 'channel-1::readline==6.2=0'), get_matchspec_from_index(index, 'channel-1::sqlite==3.7.13=0'), get_matchspec_from_index(index, 'channel-1::tk==8.5.13=0'), get_matchspec_from_index(index, 'channel-1::zlib==1.2.7=0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n  environment location: /Users/aaronmeurer/anaconda/envs/test\n\n\nThe following NEW packages will be INSTALLED:\n\n    python:   3.3.2-0 \n    readline: 6.2-0   \n    sqlite:   3.7.13-0\n    tk:       8.5.13-0\n    zlib:     1.2.7-0 \n\n'
        actions['UNLINK'] = actions['LINK']
        actions['LINK'] = []
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n  environment location: /Users/aaronmeurer/anaconda/envs/test\n\n\nThe following packages will be REMOVED:\n\n    python:   3.3.2-0 \n    readline: 6.2-0   \n    sqlite:   3.7.13-0\n    tk:       8.5.13-0\n    zlib:     1.2.7-0 \n\n'
        actions = defaultdict(list)
        actions.update({'LINK': [get_matchspec_from_index(index, 'channel-1::cython==0.19.1=py33_0')], 'UNLINK': [get_matchspec_from_index(index, 'channel-1::cython==0.19=py33_0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be UPDATED:\n\n    cython: 0.19-py33_0 --> 0.19.1-py33_0\n\n'
        (actions['LINK'], actions['UNLINK']) = (actions['UNLINK'], actions['LINK'])
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be DOWNGRADED:\n\n    cython: 0.19.1-py33_0 --> 0.19-py33_0\n\n'
        actions = defaultdict(list)
        actions.update({'LINK': [get_matchspec_from_index(index, 'channel-1::cython==0.19.1=py33_0'), get_matchspec_from_index(index, 'channel-1::dateutil==1.5=py33_0'), get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_0')], 'UNLINK': [get_matchspec_from_index(index, 'channel-1::cython==0.19=py33_0'), get_matchspec_from_index(index, 'channel-1::dateutil==2.1=py33_1'), get_matchspec_from_index(index, 'channel-1::pip==1.3.1=py33_1')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following NEW packages will be INSTALLED:\n\n    numpy:    1.7.1-py33_0\n\nThe following packages will be REMOVED:\n\n    pip:      1.3.1-py33_1\n\nThe following packages will be UPDATED:\n\n    cython:   0.19-py33_0  --> 0.19.1-py33_0\n\nThe following packages will be DOWNGRADED:\n\n    dateutil: 2.1-py33_1   --> 1.5-py33_0   \n\n'
        actions = defaultdict(list)
        actions.update({'LINK': [get_matchspec_from_index(index, 'channel-1::cython==0.19.1=py33_0'), get_matchspec_from_index(index, 'channel-1::dateutil==2.1=py33_1')], 'UNLINK': [get_matchspec_from_index(index, 'channel-1::cython==0.19=py33_0'), get_matchspec_from_index(index, 'channel-1::dateutil==1.5=py33_0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be UPDATED:\n\n    cython:   0.19-py33_0 --> 0.19.1-py33_0\n    dateutil: 1.5-py33_0  --> 2.1-py33_1   \n\n'
        (actions['LINK'], actions['UNLINK']) = (actions['UNLINK'], actions['LINK'])
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be DOWNGRADED:\n\n    cython:   0.19.1-py33_0 --> 0.19-py33_0\n    dateutil: 2.1-py33_1    --> 1.5-py33_0 \n\n'

def test_display_actions_show_channel_urls():
    if False:
        while True:
            i = 10
    with env_var('CONDA_SHOW_CHANNEL_URLS', 'True', stack_callback=conda_tests_ctxt_mgmt_def_pol):
        actions = defaultdict(list)
        sympy_prec = PackageRecord.from_objects(get_matchspec_from_index(index, 'channel-1::sympy==0.7.2=py27_0'))
        numpy_prec = PackageRecord.from_objects(get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py27_0'))
        numpy_prec.channel = sympy_prec.channel = Channel(None)
        actions.update({'FETCH': [sympy_prec, numpy_prec]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be downloaded:\n\n    package                    |            build\n    ---------------------------|-----------------\n    sympy-0.7.2                |           py27_0         4.2 MB  <unknown>\n    numpy-1.7.1                |           py27_0         5.7 MB  <unknown>\n    ------------------------------------------------------------\n                                           Total:         9.9 MB\n\n'
        actions = defaultdict(list)
        actions.update({'PREFIX': '/Users/aaronmeurer/anaconda/envs/test', 'SYMLINK_CONDA': ['/Users/aaronmeurer/anaconda'], 'LINK': [get_matchspec_from_index(index, 'channel-1::python==3.3.2=0'), get_matchspec_from_index(index, 'channel-1::readline==6.2=0'), get_matchspec_from_index(index, 'channel-1::sqlite==3.7.13=0'), get_matchspec_from_index(index, 'channel-1::tk==8.5.13=0'), get_matchspec_from_index(index, 'channel-1::zlib==1.2.7=0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n  environment location: /Users/aaronmeurer/anaconda/envs/test\n\n\nThe following NEW packages will be INSTALLED:\n\n    python:   3.3.2-0  channel-1\n    readline: 6.2-0    channel-1\n    sqlite:   3.7.13-0 channel-1\n    tk:       8.5.13-0 channel-1\n    zlib:     1.2.7-0  channel-1\n\n'
        actions['UNLINK'] = actions['LINK']
        actions['LINK'] = []
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n  environment location: /Users/aaronmeurer/anaconda/envs/test\n\n\nThe following packages will be REMOVED:\n\n    python:   3.3.2-0  channel-1\n    readline: 6.2-0    channel-1\n    sqlite:   3.7.13-0 channel-1\n    tk:       8.5.13-0 channel-1\n    zlib:     1.2.7-0  channel-1\n\n'
        actions = defaultdict(list)
        actions.update({'LINK': [get_matchspec_from_index(index, 'channel-1::cython==0.19.1=py33_0')], 'UNLINK': [get_matchspec_from_index(index, 'channel-1::cython==0.19=py33_0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be UPDATED:\n\n    cython: 0.19-py33_0 channel-1 --> 0.19.1-py33_0 channel-1\n\n'
        (actions['LINK'], actions['UNLINK']) = (actions['UNLINK'], actions['LINK'])
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be DOWNGRADED:\n\n    cython: 0.19.1-py33_0 channel-1 --> 0.19-py33_0 channel-1\n\n'
        actions = defaultdict(list)
        actions.update({'LINK': [get_matchspec_from_index(index, 'channel-1::cython==0.19.1=py33_0'), get_matchspec_from_index(index, 'channel-1::dateutil==1.5=py33_0'), get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_0')], 'UNLINK': [get_matchspec_from_index(index, 'channel-1::cython==0.19=py33_0'), get_matchspec_from_index(index, 'channel-1::dateutil==2.1=py33_1'), get_matchspec_from_index(index, 'channel-1::pip==1.3.1=py33_1')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following NEW packages will be INSTALLED:\n\n    numpy:    1.7.1-py33_0 channel-1\n\nThe following packages will be REMOVED:\n\n    pip:      1.3.1-py33_1 channel-1\n\nThe following packages will be UPDATED:\n\n    cython:   0.19-py33_0  channel-1 --> 0.19.1-py33_0 channel-1\n\nThe following packages will be DOWNGRADED:\n\n    dateutil: 2.1-py33_1   channel-1 --> 1.5-py33_0    channel-1\n\n'
        actions = defaultdict(list)
        actions.update({'LINK': [get_matchspec_from_index(index, 'channel-1::cython==0.19.1=py33_0'), get_matchspec_from_index(index, 'channel-1::dateutil==2.1=py33_1')], 'UNLINK': [get_matchspec_from_index(index, 'channel-1::cython==0.19=py33_0'), get_matchspec_from_index(index, 'channel-1::dateutil==1.5=py33_0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be UPDATED:\n\n    cython:   0.19-py33_0 channel-1 --> 0.19.1-py33_0 channel-1\n    dateutil: 1.5-py33_0  channel-1 --> 2.1-py33_1    channel-1\n\n'
        (actions['LINK'], actions['UNLINK']) = (actions['UNLINK'], actions['LINK'])
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be DOWNGRADED:\n\n    cython:   0.19.1-py33_0 channel-1 --> 0.19-py33_0 channel-1\n    dateutil: 2.1-py33_1    channel-1 --> 1.5-py33_0  channel-1\n\n'
        cython_prec = PackageRecord.from_objects(get_matchspec_from_index(index, 'channel-1::cython==0.19.1=py33_0'))
        dateutil_prec = PackageRecord.from_objects(get_matchspec_from_index(index, 'channel-1::dateutil==1.5=py33_0'))
        cython_prec.channel = dateutil_prec.channel = Channel('my_channel')
        actions = defaultdict(list)
        actions.update({'LINK': [cython_prec, get_matchspec_from_index(index, 'channel-1::dateutil==2.1=py33_1')], 'UNLINK': [get_matchspec_from_index(index, 'channel-1::cython==0.19=py33_0'), dateutil_prec]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be UPDATED:\n\n    cython:   0.19-py33_0 channel-1  --> 0.19.1-py33_0 my_channel\n    dateutil: 1.5-py33_0  my_channel --> 2.1-py33_1    channel-1 \n\n'
        (actions['LINK'], actions['UNLINK']) = (actions['UNLINK'], actions['LINK'])
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be DOWNGRADED:\n\n    cython:   0.19.1-py33_0 my_channel --> 0.19-py33_0 channel-1 \n    dateutil: 2.1-py33_1    channel-1  --> 1.5-py33_0  my_channel\n\n'

@pytest.mark.xfail(strict=True, reason='Not reporting link type until refactoring display_actions after txn.verify()')
def test_display_actions_link_type():
    if False:
        while True:
            i = 10
    with env_var('CONDA_SHOW_CHANNEL_URLS', 'False', stack_callback=conda_tests_ctxt_mgmt_def_pol):
        actions = defaultdict(list, {'LINK': ['cython-0.19.1-py33_0 2', 'dateutil-1.5-py33_0 2', 'numpy-1.7.1-py33_0 2', 'python-3.3.2-0 2', 'readline-6.2-0 2', 'sqlite-3.7.13-0 2', 'tk-8.5.13-0 2', 'zlib-1.2.7-0 2']})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\nThe following NEW packages will be INSTALLED:\n\n    cython:   0.19.1-py33_0 (softlink)\n    dateutil: 1.5-py33_0    (softlink)\n    numpy:    1.7.1-py33_0  (softlink)\n    python:   3.3.2-0       (softlink)\n    readline: 6.2-0         (softlink)\n    sqlite:   3.7.13-0      (softlink)\n    tk:       8.5.13-0      (softlink)\n    zlib:     1.2.7-0       (softlink)\n\n'
        actions = defaultdict(list, {'LINK': ['cython-0.19.1-py33_0 2', 'dateutil-2.1-py33_1 2'], 'UNLINK': ['cython-0.19-py33_0', 'dateutil-1.5-py33_0']})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\nThe following packages will be UPDATED:\n\n    cython:   0.19-py33_0 --> 0.19.1-py33_0 (softlink)\n    dateutil: 1.5-py33_0  --> 2.1-py33_1    (softlink)\n\n'
        actions = defaultdict(list, {'LINK': ['cython-0.19-py33_0 2', 'dateutil-1.5-py33_0 2'], 'UNLINK': ['cython-0.19.1-py33_0', 'dateutil-2.1-py33_1']})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\nThe following packages will be DOWNGRADED:\n\n    cython:   0.19.1-py33_0 --> 0.19-py33_0 (softlink)\n    dateutil: 2.1-py33_1    --> 1.5-py33_0  (softlink)\n\n'
        actions = defaultdict(list, {'LINK': ['cython-0.19.1-py33_0 1', 'dateutil-1.5-py33_0 1', 'numpy-1.7.1-py33_0 1', 'python-3.3.2-0 1', 'readline-6.2-0 1', 'sqlite-3.7.13-0 1', 'tk-8.5.13-0 1', 'zlib-1.2.7-0 1']})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\nThe following NEW packages will be INSTALLED:\n\n    cython:   0.19.1-py33_0\n    dateutil: 1.5-py33_0   \n    numpy:    1.7.1-py33_0 \n    python:   3.3.2-0      \n    readline: 6.2-0        \n    sqlite:   3.7.13-0     \n    tk:       8.5.13-0     \n    zlib:     1.2.7-0      \n\n'
        actions = defaultdict(list, {'LINK': ['cython-0.19.1-py33_0 1', 'dateutil-2.1-py33_1 1'], 'UNLINK': ['cython-0.19-py33_0', 'dateutil-1.5-py33_0']})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\nThe following packages will be UPDATED:\n\n    cython:   0.19-py33_0 --> 0.19.1-py33_0\n    dateutil: 1.5-py33_0  --> 2.1-py33_1   \n\n'
        actions = defaultdict(list, {'LINK': ['cython-0.19-py33_0 1', 'dateutil-1.5-py33_0 1'], 'UNLINK': ['cython-0.19.1-py33_0', 'dateutil-2.1-py33_1']})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\nThe following packages will be DOWNGRADED:\n\n    cython:   0.19.1-py33_0 --> 0.19-py33_0\n    dateutil: 2.1-py33_1    --> 1.5-py33_0 \n\n'
        actions = defaultdict(list, {'LINK': ['cython-0.19.1-py33_0 3', 'dateutil-1.5-py33_0 3', 'numpy-1.7.1-py33_0 3', 'python-3.3.2-0 3', 'readline-6.2-0 3', 'sqlite-3.7.13-0 3', 'tk-8.5.13-0 3', 'zlib-1.2.7-0 3']})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\nThe following NEW packages will be INSTALLED:\n\n    cython:   0.19.1-py33_0 (copy)\n    dateutil: 1.5-py33_0    (copy)\n    numpy:    1.7.1-py33_0  (copy)\n    python:   3.3.2-0       (copy)\n    readline: 6.2-0         (copy)\n    sqlite:   3.7.13-0      (copy)\n    tk:       8.5.13-0      (copy)\n    zlib:     1.2.7-0       (copy)\n\n'
        actions = defaultdict(list, {'LINK': ['cython-0.19.1-py33_0 3', 'dateutil-2.1-py33_1 3'], 'UNLINK': ['cython-0.19-py33_0', 'dateutil-1.5-py33_0']})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\nThe following packages will be UPDATED:\n\n    cython:   0.19-py33_0 --> 0.19.1-py33_0 (copy)\n    dateutil: 1.5-py33_0  --> 2.1-py33_1    (copy)\n\n'
        actions = defaultdict(list, {'LINK': ['cython-0.19-py33_0 3', 'dateutil-1.5-py33_0 3'], 'UNLINK': ['cython-0.19.1-py33_0', 'dateutil-2.1-py33_1']})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\nThe following packages will be DOWNGRADED:\n\n    cython:   0.19.1-py33_0 --> 0.19-py33_0 (copy)\n    dateutil: 2.1-py33_1    --> 1.5-py33_0  (copy)\n\n'
    with env_var('CONDA_SHOW_CHANNEL_URLS', 'True', stack_callback=conda_tests_ctxt_mgmt_def_pol):
        d = Dist('cython-0.19.1-py33_0.tar.bz2')
        index[d] = PackageRecord.from_objects(index[d], channel='my_channel')
        d = Dist('dateutil-1.5-py33_0.tar.bz2')
        index[d] = PackageRecord.from_objects(index[d], channel='my_channel')
        actions = defaultdict(list, {'LINK': ['cython-0.19.1-py33_0 3', 'dateutil-1.5-py33_0 3', 'numpy-1.7.1-py33_0 3', 'python-3.3.2-0 3', 'readline-6.2-0 3', 'sqlite-3.7.13-0 3', 'tk-8.5.13-0 3', 'zlib-1.2.7-0 3']})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\nThe following NEW packages will be INSTALLED:\n\n    cython:   0.19.1-py33_0 my_channel (copy)\n    dateutil: 1.5-py33_0    my_channel (copy)\n    numpy:    1.7.1-py33_0  <unknown>  (copy)\n    python:   3.3.2-0       <unknown>  (copy)\n    readline: 6.2-0         <unknown>  (copy)\n    sqlite:   3.7.13-0      <unknown>  (copy)\n    tk:       8.5.13-0      <unknown>  (copy)\n    zlib:     1.2.7-0       <unknown>  (copy)\n\n'
        actions = defaultdict(list, {'LINK': ['cython-0.19.1-py33_0 3', 'dateutil-2.1-py33_1 3'], 'UNLINK': ['cython-0.19-py33_0', 'dateutil-1.5-py33_0']})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\nThe following packages will be UPDATED:\n\n    cython:   0.19-py33_0 <unknown>  --> 0.19.1-py33_0 my_channel (copy)\n    dateutil: 1.5-py33_0  my_channel --> 2.1-py33_1    <unknown>  (copy)\n\n'
        actions = defaultdict(list, {'LINK': ['cython-0.19-py33_0 3', 'dateutil-1.5-py33_0 3'], 'UNLINK': ['cython-0.19.1-py33_0', 'dateutil-2.1-py33_1']})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\nThe following packages will be DOWNGRADED:\n\n    cython:   0.19.1-py33_0 my_channel --> 0.19-py33_0 <unknown>  (copy)\n    dateutil: 2.1-py33_1    <unknown>  --> 1.5-py33_0  my_channel (copy)\n\n'

def test_display_actions_features():
    if False:
        for i in range(10):
            print('nop')
    with env_var('CONDA_SHOW_CHANNEL_URLS', 'False', stack_callback=conda_tests_ctxt_mgmt_def_pol):
        actions = defaultdict(list)
        actions.update({'LINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_p0'), get_matchspec_from_index(index, 'channel-1::cython==0.19=py33_0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following NEW packages will be INSTALLED:\n\n    cython: 0.19-py33_0  \n    numpy:  1.7.1-py33_p0 [mkl]\n\n'
        actions = defaultdict(list)
        actions.update({'UNLINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_p0'), get_matchspec_from_index(index, 'channel-1::cython==0.19=py33_0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be REMOVED:\n\n    cython: 0.19-py33_0  \n    numpy:  1.7.1-py33_p0 [mkl]\n\n'
        actions = defaultdict(list)
        actions.update({'UNLINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_p0')], 'LINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.0=py33_p0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be DOWNGRADED:\n\n    numpy: 1.7.1-py33_p0 [mkl] --> 1.7.0-py33_p0 [mkl]\n\n'
        actions = defaultdict(list)
        actions.update({'LINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_p0')], 'UNLINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.0=py33_p0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be UPDATED:\n\n    numpy: 1.7.0-py33_p0 [mkl] --> 1.7.1-py33_p0 [mkl]\n\n'
        actions = defaultdict(list)
        actions.update({'LINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_p0')], 'UNLINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be UPDATED:\n\n    numpy: 1.7.1-py33_0 --> 1.7.1-py33_p0 [mkl]\n\n'
        actions = defaultdict(list)
        actions.update({'UNLINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_p0')], 'LINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be UPDATED:\n\n    numpy: 1.7.1-py33_p0 [mkl] --> 1.7.1-py33_0\n\n'
    with env_var('CONDA_SHOW_CHANNEL_URLS', 'True', stack_callback=conda_tests_ctxt_mgmt_def_pol):
        actions = defaultdict(list)
        actions.update({'LINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_p0'), get_matchspec_from_index(index, 'channel-1::cython==0.19=py33_0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following NEW packages will be INSTALLED:\n\n    cython: 0.19-py33_0   channel-1\n    numpy:  1.7.1-py33_p0 channel-1 [mkl]\n\n'
        actions = defaultdict(list)
        actions.update({'UNLINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_p0'), get_matchspec_from_index(index, 'channel-1::cython==0.19=py33_0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be REMOVED:\n\n    cython: 0.19-py33_0   channel-1\n    numpy:  1.7.1-py33_p0 channel-1 [mkl]\n\n'
        actions = defaultdict(list)
        actions.update({'UNLINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_p0')], 'LINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.0=py33_p0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be DOWNGRADED:\n\n    numpy: 1.7.1-py33_p0 channel-1 [mkl] --> 1.7.0-py33_p0 channel-1 [mkl]\n\n'
        actions = defaultdict(list)
        actions.update({'LINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_p0')], 'UNLINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.0=py33_p0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be UPDATED:\n\n    numpy: 1.7.0-py33_p0 channel-1 [mkl] --> 1.7.1-py33_p0 channel-1 [mkl]\n\n'
        actions = defaultdict(list)
        actions.update({'LINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_p0')], 'UNLINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be UPDATED:\n\n    numpy: 1.7.1-py33_0 channel-1 --> 1.7.1-py33_p0 channel-1 [mkl]\n\n'
        actions = defaultdict(list)
        actions.update({'UNLINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_p0')], 'LINK': [get_matchspec_from_index(index, 'channel-1::numpy==1.7.1=py33_0')]})
        with captured() as c:
            display_actions(actions, index)
        assert c.stdout == '\n## Package Plan ##\n\n\nThe following packages will be UPDATED:\n\n    numpy: 1.7.1-py33_p0 channel-1 [mkl] --> 1.7.1-py33_0 channel-1\n\n'

def test_update_old_plan():
    if False:
        for i in range(10):
            print('nop')
    old_plan = ['# plan', 'INSTRUCTION arg']
    new_plan = update_old_plan(old_plan)
    expected = [('INSTRUCTION', 'arg')]
    assert new_plan == expected
    with pytest.raises(CondaError):
        update_old_plan(['INVALID'])

def test_execute_plan(monkeypatch: MonkeyPatch):
    if False:
        i = 10
        return i + 15

    def INSTRUCTION_CMD(state, arg):
        if False:
            for i in range(10):
                print('nop')
        INSTRUCTION_CMD.called = True
        INSTRUCTION_CMD.arg = arg
    monkeypatch.setitem(inst.commands, 'INSTRUCTION', INSTRUCTION_CMD)
    old_plan = ['# plan', 'INSTRUCTION arg']
    execute_plan(old_plan)
    assert INSTRUCTION_CMD.called
    assert INSTRUCTION_CMD.arg == 'arg'

def generate_mocked_resolve(pkgs, install=None):
    if False:
        for i in range(10):
            print('nop')
    mock_package = namedtuple('PacakgeRecord', ['preferred_env', 'name', 'schannel', 'version', 'fn'])
    mock_resolve = namedtuple('Resolve', ['get_dists_for_spec', 'index', 'explicit', 'install', 'package_name', 'dependency_sort'])
    index = {}
    groups = defaultdict(list)
    for (preferred_env, name, schannel, version) in pkgs:
        dist = Dist.from_string(f'{name}-{version}-0', channel_override=schannel)
        pkg = mock_package(preferred_env=preferred_env, name=name, schannel=schannel, version=version, fn=name)
        groups[name].append(dist)
        index[dist] = pkg

    def get_dists_for_spec(spec, emptyok=False):
        if False:
            print('Hello World!')
        res = groups[spec.name]
        if not res and (not emptyok):
            raise PackagesNotFoundError((spec,))
        return res

    def get_explicit(spec):
        if False:
            while True:
                i = 10
        return True

    def get_install(spec, installed, update_deps=None):
        if False:
            while True:
                i = 10
        return install

    def get_package_name(dist):
        if False:
            return 10
        return dist.name

    def get_dependency_sort(specs):
        if False:
            for i in range(10):
                print('nop')
        return tuple((spec for spec in specs.values()))
    return mock_resolve(get_dists_for_spec=get_dists_for_spec, index=index, explicit=get_explicit, install=get_install, package_name=get_package_name, dependency_sort=get_dependency_sort)

def generate_mocked_record(dist_name):
    if False:
        i = 10
        return i + 15
    mocked_record = namedtuple('Record', ['dist_name'])
    return mocked_record(dist_name=dist_name)

def generate_mocked_context(prefix, root_prefix, envs_dirs):
    if False:
        while True:
            i = 10
    mocked_context = namedtuple('Context', ['prefix', 'root_prefix', 'envs_dirs', 'prefix_specified'])
    return mocked_context(prefix=prefix, root_prefix=root_prefix, envs_dirs=envs_dirs, prefix_specified=False)

def generate_remove_action(prefix, unlink):
    if False:
        return 10
    action = defaultdict(list)
    action['op_order'] = ('CHECK_FETCH', 'RM_FETCHED', 'FETCH', 'CHECK_EXTRACT', 'RM_EXTRACTED', 'EXTRACT', 'UNLINK', 'LINK', 'SYMLINK_CONDA')
    action['PREFIX'] = prefix
    action['UNLINK'] = unlink
    return action

def test_pinned_specs_CONDA_PINNED_PACKAGES():
    if False:
        for i in range(10):
            print('nop')
    specs = ('numpy 1.11', 'python >3')
    with env_var('CONDA_PINNED_PACKAGES', '&'.join(specs), stack_callback=conda_tests_ctxt_mgmt_def_pol):
        pinned_specs = get_pinned_specs('/none')
        assert pinned_specs != specs
        assert pinned_specs == tuple((MatchSpec(spec, optional=True) for spec in specs))

def test_pinned_specs_conda_meta_pinned(tmp_env: TmpEnvFixture):
    if False:
        i = 10
        return i + 15
    specs = ('scipy ==0.14.2', 'openjdk >=8')
    with tmp_env() as prefix:
        (prefix / 'conda-meta' / 'pinned').write_text('\n'.join(specs) + '\n')
        pinned_specs = get_pinned_specs(prefix)
        assert pinned_specs != specs
        assert pinned_specs == tuple((MatchSpec(spec, optional=True) for spec in specs))

def test_pinned_specs_condarc(tmp_env: TmpEnvFixture, conda_cli: CondaCLIFixture, mocker: MockerFixture):
    if False:
        while True:
            i = 10
    specs = ('requests ==2.13',)
    with tmp_env() as prefix:
        mocker.patch('conda.base.context.Context.active_prefix', new_callable=mocker.PropertyMock, return_value=str(prefix))
        conda_cli('config', '--env', '--add', 'pinned_packages', *specs)
        pinned_specs = get_pinned_specs(prefix)
        assert pinned_specs != specs
        assert pinned_specs == tuple((MatchSpec(spec, optional=True) for spec in specs))

def test_pinned_specs_all(tmp_env: TmpEnvFixture, conda_cli: CondaCLIFixture, mocker: MockerFixture):
    if False:
        print('Hello World!')
    specs1 = ('numpy 1.11', 'python >3')
    specs2 = ('scipy ==0.14.2', 'openjdk >=8')
    specs3 = ('requests=2.13',)
    specs = (*specs1, *specs3, *specs2)
    with tmp_env() as prefix, env_var('CONDA_PINNED_PACKAGES', '&'.join(specs1), stack_callback=conda_tests_ctxt_mgmt_def_pol):
        (prefix / 'conda-meta' / 'pinned').write_text('\n'.join(specs2) + '\n')
        mocker.patch('conda.base.context.Context.active_prefix', new_callable=mocker.PropertyMock, return_value=str(prefix))
        conda_cli('config', '--env', '--add', 'pinned_packages', *specs3)
        pinned_specs = get_pinned_specs(prefix)
        assert pinned_specs != specs
        assert pinned_specs == tuple((MatchSpec(spec, optional=True) for spec in specs))