from boltons.tbutils import ParsedException

def test_parsed_exc_basic():
    if False:
        print('Hello World!')
    _tb_str = u'Traceback (most recent call last):\n  File "example.py", line 2, in <module>\n    plarp\nNameError: name \'plarp\' is not defined'
    parsed_tb = ParsedException.from_string(_tb_str)
    print(parsed_tb)
    assert parsed_tb.exc_type == 'NameError'
    assert parsed_tb.exc_msg == "name 'plarp' is not defined"
    assert parsed_tb.frames == [{'source_line': u'plarp', 'filepath': u'example.py', 'lineno': u'2', 'funcname': u'<module>'}]
    assert parsed_tb.to_string() == _tb_str

def test_parsed_exc_nosrcline():
    if False:
        return 10
    'just making sure that everything can be parsed even if there is\n    a line without source and also if the exception has no message'
    _tb_str = u'Traceback (most recent call last):\n  File "/home/mahmoud/virtualenvs/chert/bin/chert", line 9, in <module>\n    load_entry_point(\'chert==0.2.1.dev0\', \'console_scripts\', \'chert\')()\n  File "/home/mahmoud/projects/chert/chert/core.py", line 1281, in main\n    ch.process()\n  File "/home/mahmoud/projects/chert/chert/core.py", line 741, in process\n    self.load()\n  File "<boltons.FunctionBuilder-0>", line 2, in load\n  File "/home/mahmoud/projects/lithoxyl/lithoxyl/logger.py", line 291, in logged_func\n    return func_to_log(*a, **kw)\n  File "/home/mahmoud/projects/chert/chert/core.py", line 775, in load\n    raise RuntimeError\nRuntimeError'
    parsed_tb = ParsedException.from_string(_tb_str)
    assert parsed_tb.exc_type == 'RuntimeError'
    assert parsed_tb.exc_msg == ''
    assert len(parsed_tb.frames) == 6
    assert parsed_tb.frames[3] == {'source_line': u'', 'filepath': u'<boltons.FunctionBuilder-0>', 'lineno': u'2', 'funcname': u'load'}
    assert parsed_tb.to_string() == _tb_str

def test_parsed_exc_with_anchor():
    if False:
        print('Hello World!')
    'parse a traceback with anchor lines beneath source lines'
    _tb_str = u'Traceback (most recent call last):\n  File "main.py", line 3, in <module>\n    print(add(1, "two"))\n          ^^^^^^^^^^^^^\n  File "add.py", line 2, in add\n    return a + b\n           ~~^~~\nTypeError: unsupported operand type(s) for +: \'int\' and \'str\''
    parsed_tb = ParsedException.from_string(_tb_str)
    assert parsed_tb.exc_type == 'TypeError'
    assert parsed_tb.exc_msg == "unsupported operand type(s) for +: 'int' and 'str'"
    assert parsed_tb.frames == [{'source_line': u'print(add(1, "two"))', 'filepath': u'main.py', 'lineno': u'3', 'funcname': u'<module>'}, {'source_line': u'return a + b', 'filepath': u'add.py', 'lineno': u'2', 'funcname': u'add'}]
    _tb_str_lines = _tb_str.splitlines()
    _tb_str_without_anchor = '\n'.join(_tb_str_lines[:3] + _tb_str_lines[4:6] + _tb_str_lines[7:])
    assert parsed_tb.to_string() == _tb_str_without_anchor