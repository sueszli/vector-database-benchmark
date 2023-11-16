"""
Test functions in state.py that are not a part of a class
"""
import pytest
import salt.state
from salt.utils.odict import OrderedDict
pytestmark = [pytest.mark.core_test]

def test_state_args():
    if False:
        i = 10
        return i + 15
    '\n    Testing state.state_args when this state is being used:\n\n    /etc/foo.conf:\n      file.managed:\n        - contents: "blah"\n        - mkdirs: True\n        - user: ch3ll\n        - group: ch3ll\n        - mode: 755\n\n    /etc/bar.conf:\n      file.managed:\n        - use:\n          - file: /etc/foo.conf\n    '
    id_ = '/etc/bar.conf'
    state = 'file'
    high = OrderedDict([('/etc/foo.conf', OrderedDict([('file', [OrderedDict([('contents', 'blah')]), OrderedDict([('mkdirs', True)]), OrderedDict([('user', 'ch3ll')]), OrderedDict([('group', 'ch3ll')]), OrderedDict([('mode', 755)]), 'managed', {'order': 10000}]), ('__sls__', 'test'), ('__env__', 'base')])), ('/etc/bar.conf', OrderedDict([('file', [OrderedDict([('use', [OrderedDict([('file', '/etc/foo.conf')])])]), 'managed', {'order': 10001}]), ('__sls__', 'test'), ('__env__', 'base')]))])
    ret = salt.state.state_args(id_, state, high)
    assert ret == {'order', 'use'}

def test_state_args_id_not_high():
    if False:
        return 10
    '\n    Testing state.state_args when id_ is not in high\n    '
    id_ = '/etc/bar.conf2'
    state = 'file'
    high = OrderedDict([('/etc/foo.conf', OrderedDict([('file', [OrderedDict([('contents', 'blah')]), OrderedDict([('mkdirs', True)]), OrderedDict([('user', 'ch3ll')]), OrderedDict([('group', 'ch3ll')]), OrderedDict([('mode', 755)]), 'managed', {'order': 10000}]), ('__sls__', 'test'), ('__env__', 'base')])), ('/etc/bar.conf', OrderedDict([('file', [OrderedDict([('use', [OrderedDict([('file', '/etc/foo.conf')])])]), 'managed', {'order': 10001}]), ('__sls__', 'test'), ('__env__', 'base')]))])
    ret = salt.state.state_args(id_, state, high)
    assert ret == set()

def test_state_args_state_not_high():
    if False:
        while True:
            i = 10
    '\n    Testing state.state_args when state is not in high date\n    '
    id_ = '/etc/bar.conf'
    state = 'file2'
    high = OrderedDict([('/etc/foo.conf', OrderedDict([('file', [OrderedDict([('contents', 'blah')]), OrderedDict([('mkdirs', True)]), OrderedDict([('user', 'ch3ll')]), OrderedDict([('group', 'ch3ll')]), OrderedDict([('mode', 755)]), 'managed', {'order': 10000}]), ('__sls__', 'test'), ('__env__', 'base')])), ('/etc/bar.conf', OrderedDict([('file', [OrderedDict([('use', [OrderedDict([('file', '/etc/foo.conf')])])]), 'managed', {'order': 10001}]), ('__sls__', 'test'), ('__env__', 'base')]))])
    ret = salt.state.state_args(id_, state, high)
    assert ret == set()