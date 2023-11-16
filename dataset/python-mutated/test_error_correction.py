from textwrap import dedent

def test_error_correction_with(Script):
    if False:
        for i in range(10):
            print('nop')
    source = '\n    with open() as f:\n        try:\n            f.'
    comps = Script(source).complete()
    assert len(comps) > 30
    assert [1 for c in comps if c.name == 'closed']

def test_string_literals(Script):
    if False:
        i = 10
        return i + 15
    'Simplified case of jedi-vim#377.'
    source = dedent("\n    x = ur'''\n\n    def foo():\n        pass\n    ")
    script = Script(dedent(source))
    assert script._get_module_context().tree_node.end_pos == (6, 0)
    assert not script.complete()

def test_incomplete_function(Script):
    if False:
        i = 10
        return i + 15
    source = 'return ImportErr'
    script = Script(dedent(source))
    assert script.complete(1, 3)

def test_decorator_string_issue(Script):
    if False:
        print('Hello World!')
    '\n    Test case from #589\n    '
    source = dedent('    """\n      @"""\n    def bla():\n      pass\n\n    bla.')
    s = Script(source)
    assert s.complete()
    assert s._get_module_context().tree_node.get_code() == source