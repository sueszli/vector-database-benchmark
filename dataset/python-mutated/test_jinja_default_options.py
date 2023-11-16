"""
Tests for salt.utils.jinja
"""
import os
import pytest
import salt.utils.dateutils
import salt.utils.files
import salt.utils.json
import salt.utils.stringutils
import salt.utils.yaml
from salt.utils.templates import render_jinja_tmpl

@pytest.fixture
def minion_opts(tmp_path, minion_opts):
    if False:
        while True:
            i = 10
    minion_opts.update({'cachedir': str(tmp_path / 'jinja-template-cache'), 'file_buffer_size': 1048576, 'file_client': 'local', 'file_ignore_regex': None, 'file_ignore_glob': None, 'file_roots': {'test': [str(tmp_path / 'templates')]}, 'pillar_roots': {'test': [str(tmp_path / 'templates')]}, 'fileserver_backend': ['roots'], 'hash_type': 'md5', 'extension_modules': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extmods'), 'jinja_env': {'line_comment_prefix': '##', 'line_statement_prefix': '%'}})
    return minion_opts

@pytest.fixture
def local_salt():
    if False:
        for i in range(10):
            print('nop')
    return {'myvar': 'zero', 'mylist': [0, 1, 2, 3]}

def test_comment_prefix(minion_opts, local_salt):
    if False:
        i = 10
        return i + 15
    template = "\n        %- set myvar = 'one'\n        ## ignored comment 1\n        {{- myvar -}}\n        {%- set myvar = 'two' %} ## ignored comment 2\n        {{- myvar }} ## ignored comment 3\n        %- if myvar == 'two':\n        %- set myvar = 'three'\n        %- endif\n        {{- myvar -}}\n        "
    rendered = render_jinja_tmpl(template, dict(opts=minion_opts, saltenv='test', salt=local_salt))
    assert rendered == 'onetwothree'

def test_statement_prefix(minion_opts, local_salt):
    if False:
        return 10
    template = "\n        {%- set mylist = ['1', '2', '3'] %}\n        %- set mylist = ['one', 'two', 'three']\n        %- for item in mylist:\n        {{- item }}\n        %- endfor\n        "
    rendered = render_jinja_tmpl(template, dict(opts=minion_opts, saltenv='test', salt=local_salt))
    assert rendered == 'onetwothree'