import io
import textwrap
import pytest
from scripts import validate_docstrings

class BadDocstrings:
    """Everything here has a bad docstring"""

    def private_classes(self):
        if False:
            return 10
        '\n        This mentions NDFrame, which is not correct.\n        '

    def prefix_pandas(self):
        if False:
            while True:
                i = 10
        '\n        Have `pandas` prefix in See Also section.\n\n        See Also\n        --------\n        pandas.Series.rename : Alter Series index labels or name.\n        DataFrame.head : The first `n` rows of the caller object.\n        '

    def redundant_import(self, paramx=None, paramy=None):
        if False:
            print('Hello World!')
        "\n        A sample DataFrame method.\n\n        Should not import numpy and pandas.\n\n        Examples\n        --------\n        >>> import numpy as np\n        >>> import pandas as pd\n        >>> df = pd.DataFrame(np.ones((3, 3)),\n        ...                   columns=('a', 'b', 'c'))\n        >>> df.all(1)\n        0    True\n        1    True\n        2    True\n        dtype: bool\n        >>> df.all(bool_only=True)\n        Series([], dtype: bool)\n        "

    def unused_import(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Examples\n        --------\n        >>> import pandas as pdf\n        >>> df = pd.DataFrame(np.ones((3, 3)), columns=('a', 'b', 'c'))\n        "

    def missing_whitespace_around_arithmetic_operator(self):
        if False:
            while True:
                i = 10
        '\n        Examples\n        --------\n        >>> 2+5\n        7\n        '

    def indentation_is_not_a_multiple_of_four(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Examples\n        --------\n        >>> if 2 + 5:\n        ...   pass\n        '

    def missing_whitespace_after_comma(self):
        if False:
            i = 10
            return i + 15
        "\n        Examples\n        --------\n        >>> df = pd.DataFrame(np.ones((3,3)),columns=('a','b', 'c'))\n        "

    def write_array_like_with_hyphen_not_underscore(self):
        if False:
            while True:
                i = 10
        '\n        In docstrings, use array-like over array_like\n        '

    def leftover_files(self):
        if False:
            while True:
                i = 10
        '\n        Examples\n        --------\n        >>> import pathlib\n        >>> pathlib.Path("foo.txt").touch()\n        '

class TestValidator:

    def _import_path(self, klass=None, func=None):
        if False:
            return 10
        '\n        Build the required import path for tests in this module.\n\n        Parameters\n        ----------\n        klass : str\n            Class name of object in module.\n        func : str\n            Function name of object in module.\n\n        Returns\n        -------\n        str\n            Import path of specified object in this module\n        '
        base_path = 'scripts.tests.test_validate_docstrings'
        if klass:
            base_path = f'{base_path}.{klass}'
        if func:
            base_path = f'{base_path}.{func}'
        return base_path

    def test_bad_class(self, capsys):
        if False:
            return 10
        errors = validate_docstrings.pandas_validate(self._import_path(klass='BadDocstrings'))['errors']
        assert isinstance(errors, list)
        assert errors

    @pytest.mark.parametrize('klass,func,msgs', [('BadDocstrings', 'private_classes', ('Private classes (NDFrame) should not be mentioned in public docstrings',)), ('BadDocstrings', 'prefix_pandas', ('pandas.Series.rename in `See Also` section does not need `pandas` prefix',)), ('BadDocstrings', 'redundant_import', ('Do not import numpy, as it is imported automatically',)), ('BadDocstrings', 'redundant_import', ('Do not import pandas, as it is imported automatically',)), ('BadDocstrings', 'unused_import', ("flake8 error: line 1, col 1: F401 'pandas as pdf' imported but unused",)), ('BadDocstrings', 'missing_whitespace_around_arithmetic_operator', ('flake8 error: line 1, col 2: E226 missing whitespace around arithmetic operator',)), ('BadDocstrings', 'indentation_is_not_a_multiple_of_four', ('flake8 error: line 2, col 3: E111 indentation is not a multiple of 4',)), ('BadDocstrings', 'missing_whitespace_after_comma', ("flake8 error: line 1, col 33: E231 missing whitespace after ','",)), ('BadDocstrings', 'write_array_like_with_hyphen_not_underscore', ("Use 'array-like' rather than 'array_like' in docstrings",))])
    def test_bad_docstrings(self, capsys, klass, func, msgs):
        if False:
            print('Hello World!')
        result = validate_docstrings.pandas_validate(self._import_path(klass=klass, func=func))
        for msg in msgs:
            assert msg in ' '.join([err[1] for err in result['errors']])

    def test_leftover_files_raises(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(Exception, match='The following files'):
            validate_docstrings.pandas_validate(self._import_path(klass='BadDocstrings', func='leftover_files'))

    def test_validate_all_ignore_functions(self, monkeypatch):
        if False:
            for i in range(10):
                print('nop')
        monkeypatch.setattr(validate_docstrings, 'get_all_api_items', lambda : [('pandas.DataFrame.align', 'func', 'current_section', 'current_subsection'), ('pandas.Index.all', 'func', 'current_section', 'current_subsection')])
        result = validate_docstrings.validate_all(prefix=None, ignore_functions=['pandas.DataFrame.align'])
        assert len(result) == 1
        assert 'pandas.Index.all' in result

    def test_validate_all_ignore_deprecated(self, monkeypatch):
        if False:
            return 10
        monkeypatch.setattr(validate_docstrings, 'pandas_validate', lambda func_name: {'docstring': 'docstring1', 'errors': [('ER01', 'err desc'), ('ER02', 'err desc'), ('ER03', 'err desc')], 'warnings': [], 'examples_errors': '', 'deprecated': True})
        result = validate_docstrings.validate_all(prefix=None, ignore_deprecated=True)
        assert len(result) == 0

class TestApiItems:

    @property
    def api_doc(self):
        if False:
            while True:
                i = 10
        return io.StringIO(textwrap.dedent('\n            .. currentmodule:: itertools\n\n            Itertools\n            ---------\n\n            Infinite\n            ~~~~~~~~\n\n            .. autosummary::\n\n                cycle\n                count\n\n            Finite\n            ~~~~~~\n\n            .. autosummary::\n\n                chain\n\n            .. currentmodule:: random\n\n            Random\n            ------\n\n            All\n            ~~~\n\n            .. autosummary::\n\n                seed\n                randint\n            '))

    @pytest.mark.parametrize('idx,name', [(0, 'itertools.cycle'), (1, 'itertools.count'), (2, 'itertools.chain'), (3, 'random.seed'), (4, 'random.randint')])
    def test_item_name(self, idx, name):
        if False:
            print('Hello World!')
        result = list(validate_docstrings.get_api_items(self.api_doc))
        assert result[idx][0] == name

    @pytest.mark.parametrize('idx,func', [(0, 'cycle'), (1, 'count'), (2, 'chain'), (3, 'seed'), (4, 'randint')])
    def test_item_function(self, idx, func):
        if False:
            return 10
        result = list(validate_docstrings.get_api_items(self.api_doc))
        assert callable(result[idx][1])
        assert result[idx][1].__name__ == func

    @pytest.mark.parametrize('idx,section', [(0, 'Itertools'), (1, 'Itertools'), (2, 'Itertools'), (3, 'Random'), (4, 'Random')])
    def test_item_section(self, idx, section):
        if False:
            return 10
        result = list(validate_docstrings.get_api_items(self.api_doc))
        assert result[idx][2] == section

    @pytest.mark.parametrize('idx,subsection', [(0, 'Infinite'), (1, 'Infinite'), (2, 'Finite'), (3, 'All'), (4, 'All')])
    def test_item_subsection(self, idx, subsection):
        if False:
            print('Hello World!')
        result = list(validate_docstrings.get_api_items(self.api_doc))
        assert result[idx][3] == subsection

class TestPandasDocstringClass:

    @pytest.mark.parametrize('name', ['pandas.Series.str.isdecimal', 'pandas.Series.str.islower'])
    def test_encode_content_write_to_file(self, name):
        if False:
            for i in range(10):
                print('nop')
        docstr = validate_docstrings.PandasDocstring(name).validate_pep8()
        assert not list(docstr)

class TestMainFunction:

    def test_exit_status_for_main(self, monkeypatch):
        if False:
            print('Hello World!')
        monkeypatch.setattr(validate_docstrings, 'pandas_validate', lambda func_name: {'docstring': 'docstring1', 'errors': [('ER01', 'err desc'), ('ER02', 'err desc'), ('ER03', 'err desc')], 'examples_errs': ''})
        exit_status = validate_docstrings.main(func_name='docstring1', prefix=None, errors=[], output_format='default', ignore_deprecated=False, ignore_functions=None)
        assert exit_status == 0

    def test_exit_status_errors_for_validate_all(self, monkeypatch):
        if False:
            while True:
                i = 10
        monkeypatch.setattr(validate_docstrings, 'validate_all', lambda prefix, ignore_deprecated=False, ignore_functions=None: {'docstring1': {'errors': [('ER01', 'err desc'), ('ER02', 'err desc'), ('ER03', 'err desc')], 'file': 'module1.py', 'file_line': 23}, 'docstring2': {'errors': [('ER04', 'err desc'), ('ER05', 'err desc')], 'file': 'module2.py', 'file_line': 925}})
        exit_status = validate_docstrings.main(func_name=None, prefix=None, errors=[], output_format='default', ignore_deprecated=False, ignore_functions=None)
        assert exit_status == 5

    def test_no_exit_status_noerrors_for_validate_all(self, monkeypatch):
        if False:
            print('Hello World!')
        monkeypatch.setattr(validate_docstrings, 'validate_all', lambda prefix, ignore_deprecated=False, ignore_functions=None: {'docstring1': {'errors': [], 'warnings': [('WN01', 'warn desc')]}, 'docstring2': {'errors': []}})
        exit_status = validate_docstrings.main(func_name=None, prefix=None, errors=[], output_format='default', ignore_deprecated=False, ignore_functions=None)
        assert exit_status == 0

    def test_exit_status_for_validate_all_json(self, monkeypatch):
        if False:
            for i in range(10):
                print('nop')
        print('EXECUTED')
        monkeypatch.setattr(validate_docstrings, 'validate_all', lambda prefix, ignore_deprecated=False, ignore_functions=None: {'docstring1': {'errors': [('ER01', 'err desc'), ('ER02', 'err desc'), ('ER03', 'err desc')]}, 'docstring2': {'errors': [('ER04', 'err desc'), ('ER05', 'err desc')]}})
        exit_status = validate_docstrings.main(func_name=None, prefix=None, errors=[], output_format='json', ignore_deprecated=False, ignore_functions=None)
        assert exit_status == 0

    def test_errors_param_filters_errors(self, monkeypatch):
        if False:
            return 10
        monkeypatch.setattr(validate_docstrings, 'validate_all', lambda prefix, ignore_deprecated=False, ignore_functions=None: {'Series.foo': {'errors': [('ER01', 'err desc'), ('ER02', 'err desc'), ('ER03', 'err desc')], 'file': 'series.py', 'file_line': 142}, 'DataFrame.bar': {'errors': [('ER01', 'err desc'), ('ER02', 'err desc')], 'file': 'frame.py', 'file_line': 598}, 'Series.foobar': {'errors': [('ER01', 'err desc')], 'file': 'series.py', 'file_line': 279}})
        exit_status = validate_docstrings.main(func_name=None, prefix=None, errors=['ER01'], output_format='default', ignore_deprecated=False, ignore_functions=None)
        assert exit_status == 3
        exit_status = validate_docstrings.main(func_name=None, prefix=None, errors=['ER03'], output_format='default', ignore_deprecated=False, ignore_functions=None)
        assert exit_status == 1