"""
    Test parallels desktop execution module functions
"""
import textwrap
import pytest
import salt.modules.parallels as parallels
from salt.exceptions import CommandExecutionError, SaltInvocationError
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {parallels: {}}

def test__normalize_args():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test parallels._normalize_args\n    '

    def _validate_ret(ret):
        if False:
            while True:
                i = 10
        '\n        Assert that the returned data is a list of strings\n        '
        assert isinstance(ret, list)
        for arg in ret:
            assert isinstance(arg, str)
    str_args = 'electrolytes --aqueous --anion hydroxide --cation=ammonium free radicals -- hydrogen'
    _validate_ret(parallels._normalize_args(str_args))
    list_args = ' '.join(str_args)
    _validate_ret(parallels._normalize_args(list_args))
    tuple_args = tuple(list_args)
    _validate_ret(parallels._normalize_args(tuple_args))
    other_args = {'anion': 'hydroxide', 'cation': 'ammonium'}
    _validate_ret(parallels._normalize_args(other_args))

def test__find_guids():
    if False:
        while True:
            i = 10
    '\n    Test parallels._find_guids\n    '
    guid_str = textwrap.dedent('\n        PARENT_SNAPSHOT_ID                      SNAPSHOT_ID\n                                                {a5b8999f-5d95-4aff-82de-e515b0101b66}\n        {a5b8999f-5d95-4aff-82de-e515b0101b66} *{a7345be5-ab66-478c-946e-a6c2caf14909}\n    ')
    guids = ['a5b8999f-5d95-4aff-82de-e515b0101b66', 'a7345be5-ab66-478c-946e-a6c2caf14909']
    assert parallels._find_guids(guid_str) == guids

def test_prlsrvctl():
    if False:
        i = 10
        return i + 15
    '\n    Test parallels.prlsrvctl\n    '
    runas = 'macdev'
    with patch('salt.utils.path.which', MagicMock(return_value=False)):
        with pytest.raises(CommandExecutionError):
            parallels.prlsrvctl('info', runas=runas)
    with patch('salt.utils.path.which', MagicMock(return_value='/usr/bin/prlsrvctl')):
        info_cmd = ['prlsrvctl', 'info']
        info_fcn = MagicMock()
        with patch.dict(parallels.__salt__, {'cmd.run': info_fcn}):
            parallels.prlsrvctl('info', runas=runas)
            info_fcn.assert_called_once_with(info_cmd, runas=runas)
        usb_cmd = ['prlsrvctl', 'usb', 'list']
        usb_fcn = MagicMock()
        with patch.dict(parallels.__salt__, {'cmd.run': usb_fcn}):
            parallels.prlsrvctl('usb', 'list', runas=runas)
            usb_fcn.assert_called_once_with(usb_cmd, runas=runas)
        set_cmd = ['prlsrvctl', 'set', '--mem-limit', 'auto']
        set_fcn = MagicMock()
        with patch.dict(parallels.__salt__, {'cmd.run': set_fcn}):
            parallels.prlsrvctl('set', '--mem-limit auto', runas=runas)
            set_fcn.assert_called_once_with(set_cmd, runas=runas)

def test_prlctl():
    if False:
        while True:
            i = 10
    '\n    Test parallels.prlctl\n    '
    runas = 'macdev'
    with patch('salt.utils.path.which', MagicMock(return_value=False)):
        with pytest.raises(CommandExecutionError):
            parallels.prlctl('info', runas=runas)
    with patch('salt.utils.path.which', MagicMock(return_value='/usr/bin/prlctl')):
        user_cmd = ['prlctl', 'user', 'list']
        user_fcn = MagicMock()
        with patch.dict(parallels.__salt__, {'cmd.run': user_fcn}):
            parallels.prlctl('user', 'list', runas=runas)
            user_fcn.assert_called_once_with(user_cmd, runas=runas)
        exec_cmd = ['prlctl', 'exec', 'macvm', 'uname']
        exec_fcn = MagicMock()
        with patch.dict(parallels.__salt__, {'cmd.run': exec_fcn}):
            parallels.prlctl('exec', 'macvm uname', runas=runas)
            exec_fcn.assert_called_once_with(exec_cmd, runas=runas)
        cap_cmd = ['prlctl', 'capture', 'macvm', '--file', 'macvm.display.png']
        cap_fcn = MagicMock()
        with patch.dict(parallels.__salt__, {'cmd.run': cap_fcn}):
            parallels.prlctl('capture', 'macvm --file macvm.display.png', runas=runas)
            cap_fcn.assert_called_once_with(cap_cmd, runas=runas)

def test_list_vms():
    if False:
        return 10
    '\n    Test parallels.list_vms\n    '
    runas = 'macdev'
    mock_plain = MagicMock()
    with patch.object(parallels, 'prlctl', mock_plain):
        parallels.list_vms(runas=runas)
        mock_plain.assert_called_once_with('list', [], runas=runas)
    mock_name = MagicMock()
    with patch.object(parallels, 'prlctl', mock_name):
        parallels.list_vms(name='macvm', runas=runas)
        mock_name.assert_called_once_with('list', ['macvm'], runas=runas)
    mock_templ = MagicMock()
    with patch.object(parallels, 'prlctl', mock_templ):
        parallels.list_vms(template=True, runas=runas)
        mock_templ.assert_called_once_with('list', ['--template'], runas=runas)
    mock_info = MagicMock()
    with patch.object(parallels, 'prlctl', mock_info):
        parallels.list_vms(info=True, runas=runas)
        mock_info.assert_called_once_with('list', ['--info'], runas=runas)
    mock_complex = MagicMock()
    with patch.object(parallels, 'prlctl', mock_complex):
        parallels.list_vms(args=' -o uuid,status', all=True, runas=runas)
        mock_complex.assert_called_once_with('list', ['-o', 'uuid,status', '--all'], runas=runas)

def test_clone():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test parallels.clone\n    '
    name = 'macvm'
    runas = 'macdev'
    mock_clone = MagicMock()
    with patch.object(parallels, 'prlctl', mock_clone):
        parallels.clone(name, 'macvm_new', runas=runas)
        mock_clone.assert_called_once_with('clone', [name, '--name', 'macvm_new'], runas=runas)
    mock_linked = MagicMock()
    with patch.object(parallels, 'prlctl', mock_linked):
        parallels.clone(name, 'macvm_link', linked=True, runas=runas)
        mock_linked.assert_called_once_with('clone', [name, '--name', 'macvm_link', '--linked'], runas=runas)
    mock_template = MagicMock()
    with patch.object(parallels, 'prlctl', mock_template):
        parallels.clone(name, 'macvm_templ', template=True, runas=runas)
        mock_template.assert_called_once_with('clone', [name, '--name', 'macvm_templ', '--template'], runas=runas)

def test_delete():
    if False:
        i = 10
        return i + 15
    '\n    Test parallels.delete\n    '
    name = 'macvm'
    runas = 'macdev'
    mock_delete = MagicMock()
    with patch.object(parallels, 'prlctl', mock_delete):
        parallels.delete(name, runas=runas)
        mock_delete.assert_called_once_with('delete', name, runas=runas)

def test_exists():
    if False:
        print('Hello World!')
    '\n    Test parallels.exists\n    '
    name = 'macvm'
    runas = 'macdev'
    mock_list = MagicMock(return_value='Name: {}\nState: running'.format(name))
    with patch.object(parallels, 'list_vms', mock_list):
        assert parallels.exists(name, runas=runas)
    mock_list = MagicMock(return_value='Name: {}\nState: running'.format(name))
    with patch.object(parallels, 'list_vms', mock_list):
        assert not parallels.exists('winvm', runas=runas)

def test_start():
    if False:
        while True:
            i = 10
    '\n    Test parallels.start\n    '
    name = 'macvm'
    runas = 'macdev'
    mock_start = MagicMock()
    with patch.object(parallels, 'prlctl', mock_start):
        parallels.start(name, runas=runas)
        mock_start.assert_called_once_with('start', name, runas=runas)

def test_stop():
    if False:
        return 10
    '\n    Test parallels.stop\n    '
    name = 'macvm'
    runas = 'macdev'
    mock_stop = MagicMock()
    with patch.object(parallels, 'prlctl', mock_stop):
        parallels.stop(name, runas=runas)
        mock_stop.assert_called_once_with('stop', [name], runas=runas)
    mock_kill = MagicMock()
    with patch.object(parallels, 'prlctl', mock_kill):
        parallels.stop(name, kill=True, runas=runas)
        mock_kill.assert_called_once_with('stop', [name, '--kill'], runas=runas)

def test_restart():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test parallels.restart\n    '
    name = 'macvm'
    runas = 'macdev'
    mock_start = MagicMock()
    with patch.object(parallels, 'prlctl', mock_start):
        parallels.restart(name, runas=runas)
        mock_start.assert_called_once_with('restart', name, runas=runas)

def test_reset():
    if False:
        return 10
    '\n    Test parallels.reset\n    '
    name = 'macvm'
    runas = 'macdev'
    mock_start = MagicMock()
    with patch.object(parallels, 'prlctl', mock_start):
        parallels.reset(name, runas=runas)
        mock_start.assert_called_once_with('reset', name, runas=runas)

def test_status():
    if False:
        print('Hello World!')
    '\n    Test parallels.status\n    '
    name = 'macvm'
    runas = 'macdev'
    mock_start = MagicMock()
    with patch.object(parallels, 'prlctl', mock_start):
        parallels.status(name, runas=runas)
        mock_start.assert_called_once_with('status', name, runas=runas)

def test_exec_():
    if False:
        return 10
    '\n    Test parallels.exec_\n    '
    name = 'macvm'
    runas = 'macdev'
    mock_start = MagicMock()
    with patch.object(parallels, 'prlctl', mock_start):
        parallels.exec_(name, 'find /etc/paths.d', runas=runas)
        mock_start.assert_called_once_with('exec', [name, 'find', '/etc/paths.d'], runas=runas)

def test_snapshot_id_to_name():
    if False:
        return 10
    '\n    Test parallels.snapshot_id_to_name\n    '
    name = 'macvm'
    snap_id = 'a5b8999f-5d95-4aff-82de-e515b0101b66'
    pytest.raises(SaltInvocationError, parallels.snapshot_id_to_name, name, '{8-4-4-4-12}')
    mock_no_data = MagicMock(return_value='')
    with patch.object(parallels, 'prlctl', mock_no_data):
        pytest.raises(SaltInvocationError, parallels.snapshot_id_to_name, name, snap_id)
    mock_invalid_data = MagicMock(return_value='[string theory is falsifiable}')
    with patch.object(parallels, 'prlctl', mock_invalid_data):
        snap_name = parallels.snapshot_id_to_name(name, snap_id)
        assert snap_name == ''
    mock_unknown_data = MagicMock(return_value="['sfermions', 'bosinos']")
    with patch.object(parallels, 'prlctl', mock_unknown_data):
        snap_name = parallels.snapshot_id_to_name(name, snap_id)
        assert snap_name == ''
    mock_no_name = MagicMock(return_value='Name:')
    with patch.object(parallels, 'prlctl', mock_no_name):
        snap_name = parallels.snapshot_id_to_name(name, snap_id)
        assert snap_name == ''
    mock_no_name = MagicMock(return_value='Name:')
    with patch.object(parallels, 'prlctl', mock_no_name):
        pytest.raises(SaltInvocationError, parallels.snapshot_id_to_name, name, snap_id, strict=True)
    mock_yes_name = MagicMock(return_value='Name: top')
    with patch.object(parallels, 'prlctl', mock_yes_name):
        snap_name = parallels.snapshot_id_to_name(name, snap_id)
        assert snap_name == 'top'

def test_snapshot_name_to_id():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test parallels.snapshot_name_to_id\n    '
    name = 'macvm'
    snap_ids = ['a5b8999f-5d95-4aff-82de-e515b0101b66', 'a7345be5-ab66-478c-946e-a6c2caf14909']
    snap_id = snap_ids[0]
    guid_str = textwrap.dedent('\n        PARENT_SNAPSHOT_ID                      SNAPSHOT_ID\n                                                {a5b8999f-5d95-4aff-82de-e515b0101b66}\n        {a5b8999f-5d95-4aff-82de-e515b0101b66} *{a7345be5-ab66-478c-946e-a6c2caf14909}\n    ')
    mock_guids = MagicMock(return_value=guid_str)
    with patch.object(parallels, 'prlctl', mock_guids):
        mock_no_names = MagicMock(return_value=[])
        with patch.object(parallels, 'snapshot_id_to_name', mock_no_names):
            pytest.raises(SaltInvocationError, parallels.snapshot_name_to_id, name, 'graviton')
    with patch.object(parallels, 'prlctl', mock_guids):
        mock_one_name = MagicMock(side_effect=['', 'ν_e'])
        with patch.object(parallels, 'snapshot_id_to_name', mock_one_name):
            assert parallels.snapshot_name_to_id(name, 'ν_e') == snap_ids[1]
    with patch.object(parallels, 'prlctl', mock_guids):
        mock_many_names = MagicMock(side_effect=['J/Ψ', 'J/Ψ'])
        with patch.object(parallels, 'snapshot_id_to_name', mock_many_names):
            assert sorted(parallels.snapshot_name_to_id(name, 'J/Ψ')) == sorted(snap_ids)
    with patch.object(parallels, 'prlctl', mock_guids):
        mock_many_names = MagicMock(side_effect=['J/Ψ', 'J/Ψ'])
        with patch.object(parallels, 'snapshot_id_to_name', mock_many_names):
            pytest.raises(SaltInvocationError, parallels.snapshot_name_to_id, name, 'J/Ψ', strict=True)

def test__validate_snap_name():
    if False:
        return 10
    '\n    Test parallels._validate_snap_name\n    '
    name = 'macvm'
    snap_id = 'a5b8999f-5d95-4aff-82de-e515b0101b66'
    assert parallels._validate_snap_name(name, snap_id) == snap_id
    mock_snap_symb = MagicMock(return_value=snap_id)
    with patch.object(parallels, 'snapshot_name_to_id', mock_snap_symb):
        assert parallels._validate_snap_name(name, 'π') == snap_id
        mock_snap_symb.assert_called_once_with(name, 'π', strict=True, runas=None)
    mock_snap_name = MagicMock(return_value=snap_id)
    with patch.object(parallels, 'snapshot_name_to_id', mock_snap_name):
        assert parallels._validate_snap_name(name, 'pion') == snap_id
        mock_snap_name.assert_called_once_with(name, 'pion', strict=True, runas=None)
    mock_snap_numb = MagicMock(return_value=snap_id)
    with patch.object(parallels, 'snapshot_name_to_id', mock_snap_numb):
        assert parallels._validate_snap_name(name, '3.14159') == snap_id
        mock_snap_numb.assert_called_once_with(name, '3.14159', strict=True, runas=None)
    mock_snap_non_strict = MagicMock(return_value=snap_id)
    with patch.object(parallels, 'snapshot_name_to_id', mock_snap_non_strict):
        assert parallels._validate_snap_name(name, 'e_ν', strict=False) == snap_id
        mock_snap_non_strict.assert_called_once_with(name, 'e_ν', strict=False, runas=None)

def test_list_snapshots():
    if False:
        while True:
            i = 10
    '\n    Test parallels.list_snapshots\n    '
    name = 'macvm'
    guid_str = textwrap.dedent('\n        PARENT_SNAPSHOT_ID                      SNAPSHOT_ID\n                                                {a5b8999f-5d95-4aff-82de-e515b0101b66}\n        {a5b8999f-5d95-4aff-82de-e515b0101b66} *{a7345be5-ab66-478c-946e-a6c2caf14909}\n        {a5b8999f-5d95-4aff-82de-e515b0101b66}  {5da9faef-cb0e-466d-9b41-e5571b62ac2a}\n    ')
    mock_prlctl = MagicMock(return_value=guid_str)
    with patch.object(parallels, 'prlctl', mock_prlctl):
        parallels.list_snapshots(name)
        mock_prlctl.assert_called_once_with('snapshot-list', [name], runas=None)
    mock_prlctl = MagicMock(return_value=guid_str)
    with patch.object(parallels, 'prlctl', mock_prlctl):
        parallels.list_snapshots(name, tree=True)
        mock_prlctl.assert_called_once_with('snapshot-list', [name, '--tree'], runas=None)
    snap_name = 'muon'
    mock_snap_name = MagicMock(return_value=snap_name)
    with patch.object(parallels, '_validate_snap_name', mock_snap_name):
        mock_prlctl = MagicMock(return_value=guid_str)
        with patch.object(parallels, 'prlctl', mock_prlctl):
            parallels.list_snapshots(name, snap_name)
            mock_prlctl.assert_called_once_with('snapshot-list', [name, '--id', snap_name], runas=None)
    snap_names = ['electron', 'muon', 'tauon']
    mock_snap_name = MagicMock(side_effect=snap_names)
    with patch.object(parallels, 'snapshot_id_to_name', mock_snap_name):
        mock_prlctl = MagicMock(return_value=guid_str)
        with patch.object(parallels, 'prlctl', mock_prlctl):
            ret = parallels.list_snapshots(name, names=True)
            for snap_name in snap_names:
                assert snap_name in ret
            mock_prlctl.assert_called_once_with('snapshot-list', [name], runas=None)

def test_snapshot():
    if False:
        while True:
            i = 10
    '\n    Test parallels.snapshot\n    '
    name = 'macvm'
    mock_snap = MagicMock(return_value='')
    with patch.object(parallels, 'prlctl', mock_snap):
        parallels.snapshot(name)
        mock_snap.assert_called_once_with('snapshot', [name], runas=None)
    snap_name = 'h_0'
    mock_snap_name = MagicMock(return_value='')
    with patch.object(parallels, 'prlctl', mock_snap_name):
        parallels.snapshot(name, snap_name)
        mock_snap_name.assert_called_once_with('snapshot', [name, '--name', snap_name], runas=None)
    snap_name = 'h_0'
    snap_desc = textwrap.dedent('The ground state particle of the higgs multiplet family of bosons')
    mock_snap_name = MagicMock(return_value='')
    with patch.object(parallels, 'prlctl', mock_snap_name):
        parallels.snapshot(name, snap_name, snap_desc)
        mock_snap_name.assert_called_once_with('snapshot', [name, '--name', snap_name, '--description', snap_desc], runas=None)

def test_delete_snapshot():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test parallels.delete_snapshot\n    '
    delete_message = 'Delete the snapshot...\nThe snapshot has been successfully deleted.'
    name = 'macvm'
    snap_name = 'kaon'
    snap_id = 'c2eab062-a635-4ccd-b9ae-998370f898b5'
    mock_snap_name = MagicMock(return_value=snap_id)
    with patch.object(parallels, '_validate_snap_name', mock_snap_name):
        mock_delete = MagicMock(return_value=delete_message)
        with patch.object(parallels, 'prlctl', mock_delete):
            ret = parallels.delete_snapshot(name, snap_name)
            assert ret == delete_message
            mock_delete.assert_called_once_with('snapshot-delete', [name, '--id', snap_id], runas=None)
    name = 'macvm'
    snap_name = 'higgs doublet'
    snap_ids = ['c2eab062-a635-4ccd-b9ae-998370f898b5', '8aca07c5-a0e1-4dcb-ba75-cb154d46d516']
    mock_snap_ids = MagicMock(return_value=snap_ids)
    with patch.object(parallels, '_validate_snap_name', mock_snap_ids):
        mock_delete = MagicMock(return_value=delete_message)
        with patch.object(parallels, 'prlctl', mock_delete):
            ret = parallels.delete_snapshot(name, snap_name, all=True)
            mock_ret = {snap_ids[0]: delete_message, snap_ids[1]: delete_message}
            assert ret == mock_ret
            mock_delete.assert_any_call('snapshot-delete', [name, '--id', snap_ids[0]], runas=None)
            mock_delete.assert_any_call('snapshot-delete', [name, '--id', snap_ids[1]], runas=None)

def test_revert_snapshot():
    if False:
        return 10
    '\n    Test parallels.revert_snapshot\n    '
    name = 'macvm'
    snap_name = 'k-bar'
    snap_id = 'c2eab062-a635-4ccd-b9ae-998370f898b5'
    mock_snap_name = MagicMock(return_value=snap_id)
    with patch.object(parallels, '_validate_snap_name', mock_snap_name):
        mock_delete = MagicMock(return_value='')
        with patch.object(parallels, 'prlctl', mock_delete):
            parallels.revert_snapshot(name, snap_name)
            mock_delete.assert_called_once_with('snapshot-switch', [name, '--id', snap_id], runas=None)