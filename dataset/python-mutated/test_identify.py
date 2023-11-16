from io import StringIO
from typing import List
from isort import Config, identify
from isort.identify import Import

def imports_in_code(code: str, **kwargs) -> List[identify.Import]:
    if False:
        print('Hello World!')
    return list(identify.imports(StringIO(code), **kwargs))

def test_top_only():
    if False:
        print('Hello World!')
    imports_in_function = '\nimport abc\n\ndef xyz():\n    import defg\n'
    assert len(imports_in_code(imports_in_function)) == 2
    assert len(imports_in_code(imports_in_function, top_only=True)) == 1
    imports_after_class = '\nimport abc\n\nclass MyObject:\n    pass\n\nimport defg\n'
    assert len(imports_in_code(imports_after_class)) == 2
    assert len(imports_in_code(imports_after_class, top_only=True)) == 1

def test_top_doc_string():
    if False:
        return 10
    assert len(imports_in_code('\n#! /bin/bash import x\n"""import abc\nfrom y import z\n"""\nimport abc\n')) == 1

def test_yield_and_raise_edge_cases():
    if False:
        return 10
    assert not imports_in_code('\nraise SomeException("Blah") \\\n    from exceptionsInfo.popitem()[1]\n')
    assert not imports_in_code('\ndef generator_function():\n    yield \\\n        from other_function()[1]\n')
    assert len(imports_in_code('\n# one\n\n# two\n\n\ndef function():\n    # three \\\n    import b\n    import a\n')) == 2
    assert len(imports_in_code('\n# one\n\n# two\n\n\ndef function():\n    raise \\\n    import b\n    import a\n')) == 1
    assert not imports_in_code('\ndef generator_function():\n    (\n     yield\n     from other_function()[1]\n    )\n')
    assert not imports_in_code('\ndef generator_function():\n    (\n    (\n    ((((\n    (((((\n    ((\n    (((\n     yield\n\n\n\n     from other_function()[1]\n    )))))))))))))\n    )))\n')
    assert len(imports_in_code('\ndef generator_function():\n    import os\n\n    yield \\\n    from other_function()[1]\n')) == 1
    assert not imports_in_code('\ndef generator_function():\n    (\n    (\n    ((((\n    (((((\n    ((\n    (((\n     yield\n')
    assert not imports_in_code('\ndef generator_function():\n    (\n    (\n    ((((\n    (((((\n    ((\n    (((\n     raise (\n')
    assert not imports_in_code('\ndef generator_function():\n    (\n    (\n    ((((\n    (((((\n    ((\n    (((\n     raise \\\n     from \\\n')
    assert len(imports_in_code('\ndef generator_function():\n    (\n    (\n    ((((\n    (((((\n    ((\n    (((\n     raise \\\n     from \\\n    import c\n\n    import abc\n    import xyz\n')) == 2

def test_complex_examples():
    if False:
        i = 10
        return i + 15
    assert len(imports_in_code('\nimport a, b, c; import n\n\nx = (\n    1,\n    2,\n    3\n)\n\nimport x\nfrom os \\\n    import path\nfrom os (\n    import path\n)\nfrom os import \\\n    path\nfrom os \\\n    import (\n        path\n    )\nfrom os import ( \\')) == 9
    assert not imports_in_code('from os import \\')
    assert imports_in_code('\nfrom os \\\n    import (\n        system') == [Import(line_number=2, indented=False, module='os', attribute='system', alias=None, cimport=False, file_path=None)]

def test_aliases():
    if False:
        return 10
    assert imports_in_code('import os as os')[0].alias == 'os'
    assert not imports_in_code('import os as os', config=Config(remove_redundant_aliases=True))[0].alias
    assert imports_in_code('from os import path as path')[0].alias == 'path'
    assert not imports_in_code('from os import path as path', config=Config(remove_redundant_aliases=True))[0].alias

def test_indented():
    if False:
        for i in range(10):
            print('nop')
    assert not imports_in_code('import os')[0].indented
    assert imports_in_code('     import os')[0].indented
    assert imports_in_code('\timport os')[0].indented