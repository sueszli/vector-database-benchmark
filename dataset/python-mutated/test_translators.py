import pytest
from unittest.mock import Mock
from collections import OrderedDict
from nbformat.v4 import new_code_cell
from .. import translators
from ..exceptions import PapermillException
from ..models import Parameter

@pytest.mark.parametrize('test_input,expected', [('foo', '"foo"'), ('{"foo": "bar"}', '"{\\"foo\\": \\"bar\\"}"'), ({'foo': 'bar'}, '{"foo": "bar"}'), ({'foo': '"bar"'}, '{"foo": "\\"bar\\""}'), ({'foo': ['bar']}, '{"foo": ["bar"]}'), ({'foo': {'bar': 'baz'}}, '{"foo": {"bar": "baz"}}'), ({'foo': {'bar': '"baz"'}}, '{"foo": {"bar": "\\"baz\\""}}'), (['foo'], '["foo"]'), (['foo', '"bar"'], '["foo", "\\"bar\\""]'), ([{'foo': 'bar'}], '[{"foo": "bar"}]'), ([{'foo': '"bar"'}], '[{"foo": "\\"bar\\""}]'), (12345, '12345'), (-54321, '-54321'), (1.2345, '1.2345'), (-5432.1, '-5432.1'), (float('nan'), "float('nan')"), (float('-inf'), "float('-inf')"), (float('inf'), "float('inf')"), (True, 'True'), (False, 'False'), (None, 'None')])
def test_translate_type_python(test_input, expected):
    if False:
        for i in range(10):
            print('nop')
    assert translators.PythonTranslator.translate(test_input) == expected

@pytest.mark.parametrize('parameters,expected', [({'foo': 'bar'}, '# Parameters\nfoo = "bar"\n'), ({'foo': True}, '# Parameters\nfoo = True\n'), ({'foo': 5}, '# Parameters\nfoo = 5\n'), ({'foo': 1.1}, '# Parameters\nfoo = 1.1\n'), ({'foo': ['bar', 'baz']}, '# Parameters\nfoo = ["bar", "baz"]\n'), ({'foo': {'bar': 'baz'}}, '# Parameters\nfoo = {"bar": "baz"}\n'), (OrderedDict([['foo', 'bar'], ['baz', ['buz']]]), '# Parameters\nfoo = "bar"\nbaz = ["buz"]\n')])
def test_translate_codify_python(parameters, expected):
    if False:
        print('Hello World!')
    assert translators.PythonTranslator.codify(parameters) == expected

@pytest.mark.parametrize('test_input,expected', [('', '#'), ('foo', '# foo'), ("['best effort']", "# ['best effort']")])
def test_translate_comment_python(test_input, expected):
    if False:
        print('Hello World!')
    assert translators.PythonTranslator.comment(test_input) == expected

@pytest.mark.parametrize('test_input,expected', [('a = 2', [Parameter('a', 'None', '2', '')]), ('a: int = 2', [Parameter('a', 'int', '2', '')]), ('a = 2 # type:int', [Parameter('a', 'int', '2', '')]), ('a = False # Nice variable a', [Parameter('a', 'None', 'False', 'Nice variable a')]), ('a: float = 2.258 # type: int Nice variable a', [Parameter('a', 'float', '2.258', 'Nice variable a')]), ("a = 'this is a string' # type: int Nice variable a", [Parameter('a', 'int', "'this is a string'", 'Nice variable a')]), ("a: List[str] = ['this', 'is', 'a', 'string', 'list'] # Nice variable a", [Parameter('a', 'List[str]', "['this', 'is', 'a', 'string', 'list']", 'Nice variable a')]), ("a: List[str] = [\n    'this', # First\n    'is',\n    'a',\n    'string',\n    'list' # Last\n] # Nice variable a", [Parameter('a', 'List[str]', "['this','is','a','string','list']", 'Nice variable a')]), ("a: List[str] = [\n    'this',\n    'is',\n    'a',\n    'string',\n    'list'\n] # Nice variable a", [Parameter('a', 'List[str]', "['this','is','a','string','list']", 'Nice variable a')]), ("a: List[str] = [\n                'this', # First\n                'is',\n\n                'a',\n                'string',\n                'list' # Last\n            ] # Nice variable a\n\n            b: float = -2.3432 # My b variable\n            ", [Parameter('a', 'List[str]', "['this','is','a','string','list']", 'Nice variable a'), Parameter('b', 'float', '-2.3432', 'My b variable')])])
def test_inspect_python(test_input, expected):
    if False:
        print('Hello World!')
    cell = new_code_cell(source=test_input)
    assert translators.PythonTranslator.inspect(cell) == expected

@pytest.mark.parametrize('test_input,expected', [('foo', '"foo"'), ('{"foo": "bar"}', '"{\\"foo\\": \\"bar\\"}"'), ({'foo': 'bar'}, 'list("foo" = "bar")'), ({'foo': '"bar"'}, 'list("foo" = "\\"bar\\"")'), ({'foo': ['bar']}, 'list("foo" = list("bar"))'), ({'foo': {'bar': 'baz'}}, 'list("foo" = list("bar" = "baz"))'), ({'foo': {'bar': '"baz"'}}, 'list("foo" = list("bar" = "\\"baz\\""))'), (['foo'], 'list("foo")'), (['foo', '"bar"'], 'list("foo", "\\"bar\\"")'), ([{'foo': 'bar'}], 'list(list("foo" = "bar"))'), ([{'foo': '"bar"'}], 'list(list("foo" = "\\"bar\\""))'), (12345, '12345'), (-54321, '-54321'), (1.2345, '1.2345'), (-5432.1, '-5432.1'), (True, 'TRUE'), (False, 'FALSE'), (None, 'NULL')])
def test_translate_type_r(test_input, expected):
    if False:
        i = 10
        return i + 15
    assert translators.RTranslator.translate(test_input) == expected

@pytest.mark.parametrize('test_input,expected', [('', '#'), ('foo', '# foo'), ("['best effort']", "# ['best effort']")])
def test_translate_comment_r(test_input, expected):
    if False:
        while True:
            i = 10
    assert translators.RTranslator.comment(test_input) == expected

@pytest.mark.parametrize('parameters,expected', [({'foo': 'bar'}, '# Parameters\nfoo = "bar"\n'), ({'foo': True}, '# Parameters\nfoo = TRUE\n'), ({'foo': 5}, '# Parameters\nfoo = 5\n'), ({'foo': 1.1}, '# Parameters\nfoo = 1.1\n'), ({'foo': ['bar', 'baz']}, '# Parameters\nfoo = list("bar", "baz")\n'), ({'foo': {'bar': 'baz'}}, '# Parameters\nfoo = list("bar" = "baz")\n'), (OrderedDict([['foo', 'bar'], ['baz', ['buz']]]), '# Parameters\nfoo = "bar"\nbaz = list("buz")\n'), ({'___foo': 5}, '# Parameters\nfoo = 5\n')])
def test_translate_codify_r(parameters, expected):
    if False:
        i = 10
        return i + 15
    assert translators.RTranslator.codify(parameters) == expected

@pytest.mark.parametrize('test_input,expected', [('foo', '"foo"'), ('{"foo": "bar"}', '"{\\"foo\\": \\"bar\\"}"'), ({'foo': 'bar'}, 'Map("foo" -> "bar")'), ({'foo': '"bar"'}, 'Map("foo" -> "\\"bar\\"")'), ({'foo': ['bar']}, 'Map("foo" -> Seq("bar"))'), ({'foo': {'bar': 'baz'}}, 'Map("foo" -> Map("bar" -> "baz"))'), ({'foo': {'bar': '"baz"'}}, 'Map("foo" -> Map("bar" -> "\\"baz\\""))'), (['foo'], 'Seq("foo")'), (['foo', '"bar"'], 'Seq("foo", "\\"bar\\"")'), ([{'foo': 'bar'}], 'Seq(Map("foo" -> "bar"))'), ([{'foo': '"bar"'}], 'Seq(Map("foo" -> "\\"bar\\""))'), (12345, '12345'), (-54321, '-54321'), (1.2345, '1.2345'), (-5432.1, '-5432.1'), (2147483648, '2147483648L'), (-2147483649, '-2147483649L'), (True, 'true'), (False, 'false'), (None, 'None')])
def test_translate_type_scala(test_input, expected):
    if False:
        for i in range(10):
            print('nop')
    assert translators.ScalaTranslator.translate(test_input) == expected

@pytest.mark.parametrize('test_input,expected', [('', '//'), ('foo', '// foo'), ("['best effort']", "// ['best effort']")])
def test_translate_comment_scala(test_input, expected):
    if False:
        for i in range(10):
            print('nop')
    assert translators.ScalaTranslator.comment(test_input) == expected

@pytest.mark.parametrize('input_name,input_value,expected', [('foo', '""', 'val foo = ""'), ('foo', '"bar"', 'val foo = "bar"'), ('foo', 'Map("foo" -> "bar")', 'val foo = Map("foo" -> "bar")')])
def test_translate_assign_scala(input_name, input_value, expected):
    if False:
        while True:
            i = 10
    assert translators.ScalaTranslator.assign(input_name, input_value) == expected

@pytest.mark.parametrize('parameters,expected', [({'foo': 'bar'}, '// Parameters\nval foo = "bar"\n'), ({'foo': True}, '// Parameters\nval foo = true\n'), ({'foo': 5}, '// Parameters\nval foo = 5\n'), ({'foo': 1.1}, '// Parameters\nval foo = 1.1\n'), ({'foo': ['bar', 'baz']}, '// Parameters\nval foo = Seq("bar", "baz")\n'), ({'foo': {'bar': 'baz'}}, '// Parameters\nval foo = Map("bar" -> "baz")\n'), (OrderedDict([['foo', 'bar'], ['baz', ['buz']]]), '// Parameters\nval foo = "bar"\nval baz = Seq("buz")\n')])
def test_translate_codify_scala(parameters, expected):
    if False:
        return 10
    assert translators.ScalaTranslator.codify(parameters) == expected

@pytest.mark.parametrize('test_input,expected', [('foo', '"foo"'), ('{"foo": "bar"}', '"{\\"foo\\": \\"bar\\"}"'), ({'foo': 'bar'}, 'new Dictionary<string,Object>{ { "foo" , "bar" } }'), ({'foo': '"bar"'}, 'new Dictionary<string,Object>{ { "foo" , "\\"bar\\"" } }'), (['foo'], 'new [] { "foo" }'), (['foo', '"bar"'], 'new [] { "foo", "\\"bar\\"" }'), ([{'foo': 'bar'}], 'new [] { new Dictionary<string,Object>{ { "foo" , "bar" } } }'), (12345, '12345'), (-54321, '-54321'), (1.2345, '1.2345'), (-5432.1, '-5432.1'), (2147483648, '2147483648L'), (-2147483649, '-2147483649L'), (True, 'true'), (False, 'false')])
def test_translate_type_csharp(test_input, expected):
    if False:
        return 10
    assert translators.CSharpTranslator.translate(test_input) == expected

@pytest.mark.parametrize('test_input,expected', [('', '//'), ('foo', '// foo'), ("['best effort']", "// ['best effort']")])
def test_translate_comment_csharp(test_input, expected):
    if False:
        i = 10
        return i + 15
    assert translators.CSharpTranslator.comment(test_input) == expected

@pytest.mark.parametrize('input_name,input_value,expected', [('foo', '""', 'var foo = "";'), ('foo', '"bar"', 'var foo = "bar";')])
def test_translate_assign_csharp(input_name, input_value, expected):
    if False:
        print('Hello World!')
    assert translators.CSharpTranslator.assign(input_name, input_value) == expected

@pytest.mark.parametrize('parameters,expected', [({'foo': 'bar'}, '// Parameters\nvar foo = "bar";\n'), ({'foo': True}, '// Parameters\nvar foo = true;\n'), ({'foo': 5}, '// Parameters\nvar foo = 5;\n'), ({'foo': 1.1}, '// Parameters\nvar foo = 1.1;\n'), ({'foo': ['bar', 'baz']}, '// Parameters\nvar foo = new [] { "bar", "baz" };\n'), ({'foo': {'bar': 'baz'}}, '// Parameters\nvar foo = new Dictionary<string,Object>{ { "bar" , "baz" } };\n')])
def test_translate_codify_csharp(parameters, expected):
    if False:
        print('Hello World!')
    assert translators.CSharpTranslator.codify(parameters) == expected

@pytest.mark.parametrize('test_input,expected', [('foo', '"foo"'), ('{"foo": "bar"}', '"{`"foo`": `"bar`"}"'), ({'foo': 'bar'}, '@{"foo" = "bar"}'), ({'foo': '"bar"'}, '@{"foo" = "`"bar`""}'), ({'foo': ['bar']}, '@{"foo" = @("bar")}'), ({'foo': {'bar': 'baz'}}, '@{"foo" = @{"bar" = "baz"}}'), ({'foo': {'bar': '"baz"'}}, '@{"foo" = @{"bar" = "`"baz`""}}'), (['foo'], '@("foo")'), (['foo', '"bar"'], '@("foo", "`"bar`"")'), ([{'foo': 'bar'}], '@(@{"foo" = "bar"})'), ([{'foo': '"bar"'}], '@(@{"foo" = "`"bar`""})'), (12345, '12345'), (-54321, '-54321'), (1.2345, '1.2345'), (-5432.1, '-5432.1'), (float('nan'), '[double]::NaN'), (float('-inf'), '[double]::NegativeInfinity'), (float('inf'), '[double]::PositiveInfinity'), (True, '$True'), (False, '$False'), (None, '$Null')])
def test_translate_type_powershell(test_input, expected):
    if False:
        i = 10
        return i + 15
    assert translators.PowershellTranslator.translate(test_input) == expected

@pytest.mark.parametrize('parameters,expected', [({'foo': 'bar'}, '# Parameters\n$foo = "bar"\n'), ({'foo': True}, '# Parameters\n$foo = $True\n'), ({'foo': 5}, '# Parameters\n$foo = 5\n'), ({'foo': 1.1}, '# Parameters\n$foo = 1.1\n'), ({'foo': ['bar', 'baz']}, '# Parameters\n$foo = @("bar", "baz")\n'), ({'foo': {'bar': 'baz'}}, '# Parameters\n$foo = @{"bar" = "baz"}\n'), (OrderedDict([['foo', 'bar'], ['baz', ['buz']]]), '# Parameters\n$foo = "bar"\n$baz = @("buz")\n')])
def test_translate_codify_powershell(parameters, expected):
    if False:
        return 10
    assert translators.PowershellTranslator.codify(parameters) == expected

@pytest.mark.parametrize('input_name,input_value,expected', [('foo', '""', '$foo = ""'), ('foo', '"bar"', '$foo = "bar"')])
def test_translate_assign_powershell(input_name, input_value, expected):
    if False:
        i = 10
        return i + 15
    assert translators.PowershellTranslator.assign(input_name, input_value) == expected

@pytest.mark.parametrize('test_input,expected', [('', '#'), ('foo', '# foo'), ("['best effort']", "# ['best effort']")])
def test_translate_comment_powershell(test_input, expected):
    if False:
        i = 10
        return i + 15
    assert translators.PowershellTranslator.comment(test_input) == expected

@pytest.mark.parametrize('test_input,expected', [('foo', '"foo"'), ('{"foo": "bar"}', '"{\\"foo\\": \\"bar\\"}"'), ({'foo': 'bar'}, '[ ("foo", "bar" :> IComparable) ] |> Map.ofList'), ({'foo': '"bar"'}, '[ ("foo", "\\"bar\\"" :> IComparable) ] |> Map.ofList'), (['foo'], '[ "foo" ]'), (['foo', '"bar"'], '[ "foo"; "\\"bar\\"" ]'), ([{'foo': 'bar'}], '[ [ ("foo", "bar" :> IComparable) ] |> Map.ofList ]'), (12345, '12345'), (-54321, '-54321'), (1.2345, '1.2345'), (-5432.1, '-5432.1'), (2147483648, '2147483648L'), (-2147483649, '-2147483649L'), (True, 'true'), (False, 'false')])
def test_translate_type_fsharp(test_input, expected):
    if False:
        i = 10
        return i + 15
    assert translators.FSharpTranslator.translate(test_input) == expected

@pytest.mark.parametrize('test_input,expected', [('', '(*  *)'), ('foo', '(* foo *)'), ("['best effort']", "(* ['best effort'] *)")])
def test_translate_comment_fsharp(test_input, expected):
    if False:
        print('Hello World!')
    assert translators.FSharpTranslator.comment(test_input) == expected

@pytest.mark.parametrize('input_name,input_value,expected', [('foo', '""', 'let foo = ""'), ('foo', '"bar"', 'let foo = "bar"')])
def test_translate_assign_fsharp(input_name, input_value, expected):
    if False:
        return 10
    assert translators.FSharpTranslator.assign(input_name, input_value) == expected

@pytest.mark.parametrize('parameters,expected', [({'foo': 'bar'}, '(* Parameters *)\nlet foo = "bar"\n'), ({'foo': True}, '(* Parameters *)\nlet foo = true\n'), ({'foo': 5}, '(* Parameters *)\nlet foo = 5\n'), ({'foo': 1.1}, '(* Parameters *)\nlet foo = 1.1\n'), ({'foo': ['bar', 'baz']}, '(* Parameters *)\nlet foo = [ "bar"; "baz" ]\n'), ({'foo': {'bar': 'baz'}}, '(* Parameters *)\nlet foo = [ ("bar", "baz" :> IComparable) ] |> Map.ofList\n')])
def test_translate_codify_fsharp(parameters, expected):
    if False:
        return 10
    assert translators.FSharpTranslator.codify(parameters) == expected

@pytest.mark.parametrize('test_input,expected', [('foo', '"foo"'), ('{"foo": "bar"}', '"{\\"foo\\": \\"bar\\"}"'), ({'foo': 'bar'}, 'Dict("foo" => "bar")'), ({'foo': '"bar"'}, 'Dict("foo" => "\\"bar\\"")'), ({'foo': ['bar']}, 'Dict("foo" => ["bar"])'), ({'foo': {'bar': 'baz'}}, 'Dict("foo" => Dict("bar" => "baz"))'), ({'foo': {'bar': '"baz"'}}, 'Dict("foo" => Dict("bar" => "\\"baz\\""))'), (['foo'], '["foo"]'), (['foo', '"bar"'], '["foo", "\\"bar\\""]'), ([{'foo': 'bar'}], '[Dict("foo" => "bar")]'), ([{'foo': '"bar"'}], '[Dict("foo" => "\\"bar\\"")]'), (12345, '12345'), (-54321, '-54321'), (1.2345, '1.2345'), (-5432.1, '-5432.1'), (True, 'true'), (False, 'false'), (None, 'nothing')])
def test_translate_type_julia(test_input, expected):
    if False:
        for i in range(10):
            print('nop')
    assert translators.JuliaTranslator.translate(test_input) == expected

@pytest.mark.parametrize('parameters,expected', [({'foo': 'bar'}, '# Parameters\nfoo = "bar"\n'), ({'foo': True}, '# Parameters\nfoo = true\n'), ({'foo': 5}, '# Parameters\nfoo = 5\n'), ({'foo': 1.1}, '# Parameters\nfoo = 1.1\n'), ({'foo': ['bar', 'baz']}, '# Parameters\nfoo = ["bar", "baz"]\n'), ({'foo': {'bar': 'baz'}}, '# Parameters\nfoo = Dict("bar" => "baz")\n'), (OrderedDict([['foo', 'bar'], ['baz', ['buz']]]), '# Parameters\nfoo = "bar"\nbaz = ["buz"]\n')])
def test_translate_codify_julia(parameters, expected):
    if False:
        return 10
    assert translators.JuliaTranslator.codify(parameters) == expected

@pytest.mark.parametrize('test_input,expected', [('', '#'), ('foo', '# foo'), ('["best effort"]', '# ["best effort"]')])
def test_translate_comment_julia(test_input, expected):
    if False:
        print('Hello World!')
    assert translators.JuliaTranslator.comment(test_input) == expected

@pytest.mark.parametrize('test_input,expected', [('foo', '"foo"'), ('{"foo": "bar"}', '"{""foo"": ""bar""}"'), ({1: 'foo'}, 'containers.Map({\'1\'}, {"foo"})'), ({1.0: 'foo'}, 'containers.Map({\'1.0\'}, {"foo"})'), ({None: 'foo'}, 'containers.Map({\'None\'}, {"foo"})'), ({True: 'foo'}, 'containers.Map({\'True\'}, {"foo"})'), ({'foo': 'bar'}, 'containers.Map({\'foo\'}, {"bar"})'), ({'foo': '"bar"'}, 'containers.Map({\'foo\'}, {"""bar"""})'), ({'foo': ['bar']}, 'containers.Map({\'foo\'}, {{"bar"}})'), ({'foo': {'bar': 'baz'}}, 'containers.Map({\'foo\'}, {containers.Map({\'bar\'}, {"baz"})})'), ({'foo': {'bar': '"baz"'}}, 'containers.Map({\'foo\'}, {containers.Map({\'bar\'}, {"""baz"""})})'), (['foo'], '{"foo"}'), (['foo', '"bar"'], '{"foo", """bar"""}'), ([{'foo': 'bar'}], '{containers.Map({\'foo\'}, {"bar"})}'), ([{'foo': '"bar"'}], '{containers.Map({\'foo\'}, {"""bar"""})}'), (12345, '12345'), (-54321, '-54321'), (1.2345, '1.2345'), (-5432.1, '-5432.1'), (True, 'true'), (False, 'false'), (None, 'NaN')])
def test_translate_type_matlab(test_input, expected):
    if False:
        print('Hello World!')
    assert translators.MatlabTranslator.translate(test_input) == expected

@pytest.mark.parametrize('parameters,expected', [({'foo': 'bar'}, '% Parameters\nfoo = "bar";\n'), ({'foo': True}, '% Parameters\nfoo = true;\n'), ({'foo': 5}, '% Parameters\nfoo = 5;\n'), ({'foo': 1.1}, '% Parameters\nfoo = 1.1;\n'), ({'foo': ['bar', 'baz']}, '% Parameters\nfoo = {"bar", "baz"};\n'), ({'foo': {'bar': 'baz'}}, '% Parameters\nfoo = containers.Map({\'bar\'}, {"baz"});\n'), (OrderedDict([['foo', 'bar'], ['baz', ['buz']]]), '% Parameters\nfoo = "bar";\nbaz = {"buz"};\n')])
def test_translate_codify_matlab(parameters, expected):
    if False:
        for i in range(10):
            print('nop')
    assert translators.MatlabTranslator.codify(parameters) == expected

@pytest.mark.parametrize('test_input,expected', [('', '%'), ('foo', '% foo'), ("['best effort']", "% ['best effort']")])
def test_translate_comment_matlab(test_input, expected):
    if False:
        i = 10
        return i + 15
    assert translators.MatlabTranslator.comment(test_input) == expected

def test_find_translator_with_exact_kernel_name():
    if False:
        return 10
    my_new_kernel_translator = Mock()
    my_new_language_translator = Mock()
    translators.papermill_translators.register('my_new_kernel', my_new_kernel_translator)
    translators.papermill_translators.register('my_new_language', my_new_language_translator)
    assert translators.papermill_translators.find_translator('my_new_kernel', 'my_new_language') is my_new_kernel_translator

def test_find_translator_with_exact_language():
    if False:
        while True:
            i = 10
    my_new_language_translator = Mock()
    translators.papermill_translators.register('my_new_language', my_new_language_translator)
    assert translators.papermill_translators.find_translator('unregistered_kernel', 'my_new_language') is my_new_language_translator

def test_find_translator_with_no_such_kernel_or_language():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(PapermillException):
        translators.papermill_translators.find_translator('unregistered_kernel', 'unregistered_language')

def test_translate_uses_str_representation_of_unknown_types():
    if False:
        print('Hello World!')

    class FooClass:

        def __str__(self):
            if False:
                return 10
            return 'foo'
    obj = FooClass()
    assert translators.Translator.translate(obj) == '"foo"'

def test_translator_must_implement_translate_dict():
    if False:
        i = 10
        return i + 15

    class MyNewTranslator(translators.Translator):
        pass
    with pytest.raises(NotImplementedError):
        MyNewTranslator.translate_dict({'foo': 'bar'})

def test_translator_must_implement_translate_list():
    if False:
        print('Hello World!')

    class MyNewTranslator(translators.Translator):
        pass
    with pytest.raises(NotImplementedError):
        MyNewTranslator.translate_list(['foo', 'bar'])

def test_translator_must_implement_comment():
    if False:
        for i in range(10):
            print('nop')

    class MyNewTranslator(translators.Translator):
        pass
    with pytest.raises(NotImplementedError):
        MyNewTranslator.comment('foo')

@pytest.mark.parametrize('test_input,expected', [('foo', 'foo'), ('foo space', "'foo space'"), ("foo's apostrophe", '\'foo\'"\'"\'s apostrophe\''), ('shell ( is ) <dumb>', "'shell ( is ) <dumb>'"), (12345, '12345'), (-54321, '-54321'), (1.2345, '1.2345'), (-5432.1, '-5432.1'), (True, 'true'), (False, 'false'), (None, '')])
def test_translate_type_sh(test_input, expected):
    if False:
        while True:
            i = 10
    assert translators.BashTranslator.translate(test_input) == expected

@pytest.mark.parametrize('test_input,expected', [('', '#'), ('foo', '# foo'), ("['best effort']", "# ['best effort']")])
def test_translate_comment_sh(test_input, expected):
    if False:
        for i in range(10):
            print('nop')
    assert translators.BashTranslator.comment(test_input) == expected

@pytest.mark.parametrize('parameters,expected', [({'foo': 'bar'}, '# Parameters\nfoo=bar\n'), ({'foo': 'shell ( is ) <dumb>'}, "# Parameters\nfoo='shell ( is ) <dumb>'\n"), ({'foo': True}, '# Parameters\nfoo=true\n'), ({'foo': 5}, '# Parameters\nfoo=5\n'), ({'foo': 1.1}, '# Parameters\nfoo=1.1\n'), (OrderedDict([['foo', 'bar'], ['baz', '$dumb(shell)']]), "# Parameters\nfoo=bar\nbaz='$dumb(shell)'\n")])
def test_translate_codify_sh(parameters, expected):
    if False:
        for i in range(10):
            print('nop')
    assert translators.BashTranslator.codify(parameters) == expected