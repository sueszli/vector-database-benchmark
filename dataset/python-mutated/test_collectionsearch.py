from __future__ import annotations
from ansible.errors import AnsibleParserError
from ansible.playbook.play import Play
from ansible.playbook.task import Task
from ansible.playbook.block import Block
import pytest

def test_collection_static_warning(capsys):
    if False:
        i = 10
        return i + 15
    'Test that collection name is not templated.\n\n    Also, make sure that users see the warning message for the referenced name.\n    '
    collection_name = 'foo.{{bar}}'
    p = Play.load(dict(name='test play', hosts=['foo'], gather_facts=False, connection='local', collections=collection_name))
    assert collection_name in p.collections
    (std_out, std_err) = capsys.readouterr()
    assert '[WARNING]: "collections" is not templatable, but we found: %s' % collection_name in std_err
    assert '' == std_out

def test_collection_invalid_data_play():
    if False:
        for i in range(10):
            print('nop')
    'Test that collection as a dict at the play level fails with parser error'
    collection_name = {'name': 'foo'}
    with pytest.raises(AnsibleParserError):
        Play.load(dict(name='test play', hosts=['foo'], gather_facts=False, connection='local', collections=collection_name))

def test_collection_invalid_data_task():
    if False:
        i = 10
        return i + 15
    'Test that collection as a dict at the task level fails with parser error'
    collection_name = {'name': 'foo'}
    with pytest.raises(AnsibleParserError):
        Task.load(dict(name='test task', collections=collection_name))

def test_collection_invalid_data_block():
    if False:
        return 10
    'Test that collection as a dict at the block level fails with parser error'
    collection_name = {'name': 'foo'}
    with pytest.raises(AnsibleParserError):
        Block.load(dict(block=[dict(name='test task', collections=collection_name)]))