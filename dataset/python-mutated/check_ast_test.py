from __future__ import annotations
from pre_commit_hooks.check_ast import main
from testing.util import get_resource_path

def test_failing_file():
    if False:
        for i in range(10):
            print('nop')
    ret = main([get_resource_path('cannot_parse_ast.notpy')])
    assert ret == 1

def test_passing_file():
    if False:
        for i in range(10):
            print('nop')
    ret = main([__file__])
    assert ret == 0