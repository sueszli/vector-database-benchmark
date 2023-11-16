from test.helpers import get_example_dir, example_dir
from jedi import Project

def test_implicit_namespace_package(Script):
    if False:
        return 10
    sys_path = [get_example_dir('implicit_namespace_package', 'ns1'), get_example_dir('implicit_namespace_package', 'ns2')]
    project = Project('.', sys_path=sys_path)

    def script_with_path(*args, **kwargs):
        if False:
            return 10
        return Script(*args, project=project, **kwargs)
    assert script_with_path('from pkg import ns1_file').infer()
    assert script_with_path('from pkg import ns2_file').infer()
    assert not script_with_path('from pkg import ns3_file').infer()
    tests = {'from pkg.ns2_file import foo': 'ns2_file!', 'from pkg.ns1_file import foo': 'ns1_file!'}
    for (source, solution) in tests.items():
        ass = script_with_path(source).goto()
        assert len(ass) == 1
        assert ass[0].description == "foo = '%s'" % solution
    completions = script_with_path('from pkg import ').complete()
    names = [c.name for c in completions]
    compare = ['ns1_file', 'ns2_file']
    assert set(compare) == set(names)
    tests = {'from pkg import ns2_file as x': 'ns2_file!', 'from pkg import ns1_file as x': 'ns1_file!'}
    for (source, solution) in tests.items():
        for c in script_with_path(source + '; x.').complete():
            if c.name == 'foo':
                completion = c
        solution = "foo = '%s'" % solution
        assert completion.description == solution
    (c,) = script_with_path('import pkg').complete()
    assert c.docstring() == ''

def test_implicit_nested_namespace_package(Script):
    if False:
        while True:
            i = 10
    code = 'from implicit_nested_namespaces.namespace.pkg.module import CONST'
    project = Project('.', sys_path=[example_dir])
    script = Script(code, project=project)
    result = script.infer(line=1, column=61)
    assert len(result) == 1
    (implicit_pkg,) = Script(code, project=project).infer(column=10)
    assert implicit_pkg.type == 'namespace'
    assert implicit_pkg.module_path is None

def test_implicit_namespace_package_import_autocomplete(Script):
    if False:
        for i in range(10):
            print('nop')
    code = 'from implicit_name'
    project = Project('.', sys_path=[example_dir])
    script = Script(code, project=project)
    compl = script.complete()
    assert [c.name for c in compl] == ['implicit_namespace_package']

def test_namespace_package_in_multiple_directories_autocompletion(Script):
    if False:
        return 10
    code = 'from pkg.'
    sys_path = [get_example_dir('implicit_namespace_package', 'ns1'), get_example_dir('implicit_namespace_package', 'ns2')]
    project = Project('.', sys_path=sys_path)
    script = Script(code, project=project)
    compl = script.complete()
    assert set((c.name for c in compl)) == set(['ns1_file', 'ns2_file'])

def test_namespace_package_in_multiple_directories_goto_definition(Script):
    if False:
        return 10
    code = 'from pkg import ns1_file'
    sys_path = [get_example_dir('implicit_namespace_package', 'ns1'), get_example_dir('implicit_namespace_package', 'ns2')]
    project = Project('.', sys_path=sys_path)
    script = Script(code, project=project)
    result = script.infer()
    assert len(result) == 1

def test_namespace_name_autocompletion_full_name(Script):
    if False:
        for i in range(10):
            print('nop')
    code = 'from pk'
    sys_path = [get_example_dir('implicit_namespace_package', 'ns1'), get_example_dir('implicit_namespace_package', 'ns2')]
    project = Project('.', sys_path=sys_path)
    script = Script(code, project=project)
    compl = script.complete()
    assert set((c.full_name for c in compl)) == set(['pkg'])