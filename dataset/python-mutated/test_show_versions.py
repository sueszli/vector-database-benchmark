import json
import os
import re
from pandas.util._print_versions import _get_dependency_info, _get_sys_info
import pandas as pd

def test_show_versions(tmpdir):
    if False:
        while True:
            i = 10
    as_json = os.path.join(tmpdir, 'test_output.json')
    pd.show_versions(as_json=as_json)
    with open(as_json, encoding='utf-8') as fd:
        result = json.load(fd)
    expected = {'system': _get_sys_info(), 'dependencies': _get_dependency_info()}
    assert result == expected

def test_show_versions_console_json(capsys):
    if False:
        print('Hello World!')
    pd.show_versions(as_json=True)
    stdout = capsys.readouterr().out
    result = json.loads(stdout)
    expected = {'system': _get_sys_info(), 'dependencies': _get_dependency_info()}
    assert result == expected

def test_show_versions_console(capsys):
    if False:
        return 10
    pd.show_versions(as_json=False)
    result = capsys.readouterr().out
    assert 'INSTALLED VERSIONS' in result
    assert re.search('commit\\s*:\\s[0-9a-f]{40}\\n', result)
    assert re.search('numpy\\s*:\\s[0-9]+\\..*\\n', result)
    assert re.search('pyarrow\\s*:\\s([0-9]+.*|None)\\n', result)

def test_json_output_match(capsys, tmpdir):
    if False:
        print('Hello World!')
    pd.show_versions(as_json=True)
    result_console = capsys.readouterr().out
    out_path = os.path.join(tmpdir, 'test_json.json')
    pd.show_versions(as_json=out_path)
    with open(out_path, encoding='utf-8') as out_fd:
        result_file = out_fd.read()
    assert result_console == result_file