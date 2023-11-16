import textwrap
import pytest
pytestmark = [pytest.mark.windows_whitelisted]

def test_issue_8343_accumulated_require_in(modules, tmp_path, state_tree):
    if False:
        i = 10
        return i + 15
    name = tmp_path / 'testfile'
    sls_contents = f"""\n    {name}:\n      file.managed:\n        - contents: |\n                    #\n\n    prepend-foo-accumulator-from-pillar:\n      file.accumulated:\n        - require_in:\n          - file: prepend-foo-management\n        - filename: {name}\n        - text: |\n                foo\n\n    append-foo-accumulator-from-pillar:\n      file.accumulated:\n        - require_in:\n          - file: append-foo-management\n        - filename: {name}\n        - text: |\n                bar\n\n    prepend-foo-management:\n      file.blockreplace:\n        - name: {name}\n        - marker_start: "#-- start salt managed zonestart -- PLEASE, DO NOT EDIT"\n        - marker_end: "#-- end salt managed zonestart --"\n        - content: ''\n        - prepend_if_not_found: True\n        - backup: '.bak'\n        - show_changes: True\n\n    append-foo-management:\n      file.blockreplace:\n        - name: {name}\n        - marker_start: "#-- start salt managed zoneend -- PLEASE, DO NOT EDIT"\n        - marker_end: "#-- end salt managed zoneend --"\n        - content: ''\n        - append_if_not_found: True\n        - backup: '.bak2'\n        - show_changes: True\n    """
    with pytest.helpers.temp_file('issue-8343.sls', directory=state_tree, contents=sls_contents):
        ret = modules.state.sls('issue-8343')
        for state_run in ret:
            assert state_run.result is True
    expected = textwrap.dedent('    #-- start salt managed zonestart -- PLEASE, DO NOT EDIT\n    foo\n    #-- end salt managed zonestart --\n    #\n    #-- start salt managed zoneend -- PLEASE, DO NOT EDIT\n    bar\n    #-- end salt managed zoneend --\n    ')
    assert name.read_text() == expected

def test_issue_11003_immutable_lazy_proxy_sum(modules, tmp_path, state_tree):
    if False:
        for i in range(10):
            print('nop')
    name = tmp_path / 'testfile'
    sls_contents = f"""\n    a{name}:\n      file.absent:\n        - name: {name}\n\n    {name}:\n      file.managed:\n        - contents: |\n                    #\n\n    test-acc1:\n      file.accumulated:\n        - require_in:\n          - file: final\n        - filename: {name}\n        - text: |\n                bar\n\n    test-acc2:\n      file.accumulated:\n        - watch_in:\n          - file: final\n        - filename: {name}\n        - text: |\n                baz\n\n    final:\n      file.blockreplace:\n        - name: {name}\n        - marker_start: "#-- start managed zone PLEASE, DO NOT EDIT"\n        - marker_end: "#-- end managed zone"\n        - content: ''\n        - append_if_not_found: True\n        - show_changes: True\n    """
    with pytest.helpers.temp_file('issue-11003.sls', directory=state_tree, contents=sls_contents):
        ret = modules.state.sls('issue-11003')
        for state_run in ret:
            assert state_run.result is True
    contents = name.read_text().splitlines()
    begin = contents.index('#-- start managed zone PLEASE, DO NOT EDIT') + 1
    end = contents.index('#-- end managed zone')
    block_contents = contents[begin:end]
    for item in ('', 'bar', 'baz'):
        block_contents.remove(item)
    assert block_contents == []