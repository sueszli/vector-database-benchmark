from __future__ import annotations
import os
import shutil
import pytest
from pre_commit_hooks.pretty_format_json import main
from pre_commit_hooks.pretty_format_json import parse_num_to_int
from testing.util import get_resource_path

def test_parse_num_to_int():
    if False:
        print('Hello World!')
    assert parse_num_to_int('0') == 0
    assert parse_num_to_int('2') == 2
    assert parse_num_to_int('\t') == '\t'
    assert parse_num_to_int('  ') == '  '

@pytest.mark.parametrize(('filename', 'expected_retval'), (('not_pretty_formatted_json.json', 1), ('unsorted_pretty_formatted_json.json', 1), ('non_ascii_pretty_formatted_json.json', 1), ('pretty_formatted_json.json', 0)))
def test_main(filename, expected_retval):
    if False:
        i = 10
        return i + 15
    ret = main([get_resource_path(filename)])
    assert ret == expected_retval

@pytest.mark.parametrize(('filename', 'expected_retval'), (('not_pretty_formatted_json.json', 1), ('unsorted_pretty_formatted_json.json', 0), ('non_ascii_pretty_formatted_json.json', 1), ('pretty_formatted_json.json', 0)))
def test_unsorted_main(filename, expected_retval):
    if False:
        while True:
            i = 10
    ret = main(['--no-sort-keys', get_resource_path(filename)])
    assert ret == expected_retval

@pytest.mark.parametrize(('filename', 'expected_retval'), (('not_pretty_formatted_json.json', 1), ('unsorted_pretty_formatted_json.json', 1), ('non_ascii_pretty_formatted_json.json', 1), ('pretty_formatted_json.json', 1), ('tab_pretty_formatted_json.json', 0)))
def test_tab_main(filename, expected_retval):
    if False:
        while True:
            i = 10
    ret = main(['--indent', '\t', get_resource_path(filename)])
    assert ret == expected_retval

def test_non_ascii_main():
    if False:
        return 10
    ret = main(('--no-ensure-ascii', get_resource_path('non_ascii_pretty_formatted_json.json')))
    assert ret == 0

def test_autofix_main(tmpdir):
    if False:
        return 10
    srcfile = tmpdir.join('to_be_json_formatted.json')
    shutil.copyfile(get_resource_path('not_pretty_formatted_json.json'), str(srcfile))
    ret = main(['--autofix', str(srcfile)])
    assert ret == 1
    ret = main([str(srcfile)])
    assert ret == 0

def test_orderfile_get_pretty_format():
    if False:
        i = 10
        return i + 15
    ret = main(('--top-keys=alist', get_resource_path('pretty_formatted_json.json')))
    assert ret == 0

def test_not_orderfile_get_pretty_format():
    if False:
        while True:
            i = 10
    ret = main(('--top-keys=blah', get_resource_path('pretty_formatted_json.json')))
    assert ret == 1

def test_top_sorted_get_pretty_format():
    if False:
        print('Hello World!')
    ret = main(('--top-keys=01-alist,alist', get_resource_path('top_sorted_json.json')))
    assert ret == 0

def test_badfile_main():
    if False:
        while True:
            i = 10
    ret = main([get_resource_path('ok_yaml.yaml')])
    assert ret == 1

def test_diffing_output(capsys):
    if False:
        for i in range(10):
            print('nop')
    resource_path = get_resource_path('not_pretty_formatted_json.json')
    expected_retval = 1
    a = os.path.join('a', resource_path)
    b = os.path.join('b', resource_path)
    expected_out = f'--- {a}\n+++ {b}\n@@ -1,6 +1,9 @@\n {{\n-    "foo":\n-    "bar",\n-        "alist": [2, 34, 234],\n-  "blah": null\n+  "alist": [\n+    2,\n+    34,\n+    234\n+  ],\n+  "blah": null,\n+  "foo": "bar"\n }}\n'
    actual_retval = main([resource_path])
    (actual_out, actual_err) = capsys.readouterr()
    assert actual_retval == expected_retval
    assert actual_out == expected_out
    assert actual_err == ''