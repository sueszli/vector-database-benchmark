"""Unit tests for the CodeManager class"""
from typing import Optional
from unittest.mock import MagicMock, Mock, patch
import pandas as pd
import pytest
from pandasai.exceptions import BadImportError, NoCodeFoundError
from pandasai.llm.fake import FakeLLM
from pandasai.smart_dataframe import SmartDataframe
from pandasai.helpers.code_manager import CodeExecutionContext, CodeManager

class TestCodeManager:
    """Unit tests for the CodeManager class"""

    @pytest.fixture
    def llm(self, output: Optional[str]=None):
        if False:
            return 10
        return FakeLLM(output=output)

    @pytest.fixture
    def sample_df(self):
        if False:
            return 10
        return pd.DataFrame({'country': ['United States', 'United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'Canada', 'Australia', 'Japan', 'China'], 'gdp': [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064], 'happiness_index': [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]})

    @pytest.fixture
    def smart_dataframe(self, llm, sample_df):
        if False:
            for i in range(10):
                print('nop')
        return SmartDataframe(sample_df, config={'llm': llm, 'enable_cache': False})

    @pytest.fixture
    def code_manager(self, smart_dataframe: SmartDataframe):
        if False:
            i = 10
            return i + 15
        return smart_dataframe.lake._code_manager

    @pytest.fixture
    def exec_context(self) -> MagicMock:
        if False:
            return 10
        context = MagicMock(spec=CodeExecutionContext)
        return context

    def test_run_code_for_calculations(self, code_manager: CodeManager, exec_context: MagicMock):
        if False:
            print('Hello World!')
        code = "def analyze_data(dfs):\n    return {'type': 'number', 'value': 1 + 1}"
        assert code_manager.execute_code(code, exec_context)['value'] == 2
        assert code_manager.last_code_executed == code

    def test_run_code_invalid_code(self, code_manager: CodeManager, exec_context: MagicMock):
        if False:
            return 10
        with pytest.raises(Exception):
            code_manager.execute_code('1+ ', exec_context)['value']

    def test_clean_code_remove_builtins(self, code_manager: CodeManager, exec_context: MagicMock):
        if False:
            while True:
                i = 10
        builtins_code = "import set\ndef analyze_data(dfs):\n    return {'type': 'number', 'value': set([1, 2, 3])}"
        assert code_manager.execute_code(builtins_code, exec_context)['value'] == {1, 2, 3}
        assert code_manager.last_code_executed == "def analyze_data(dfs):\n    return {'type': 'number', 'value': set([1, 2, 3])}"

    def test_clean_code_removes_jailbreak_code(self, code_manager: CodeManager, exec_context: MagicMock):
        if False:
            while True:
                i = 10
        malicious_code = "def analyze_data(dfs):\n    __builtins__['str'].__class__.__mro__[-1].__subclasses__()[140].__init__.__globals__['system']('ls')\n    print('hello world')"
        assert code_manager._clean_code(malicious_code, exec_context) == "def analyze_data(dfs):\n    print('hello world')"

    def test_clean_code_remove_environment_defaults(self, code_manager: CodeManager, exec_context: MagicMock):
        if False:
            return 10
        pandas_code = "import pandas as pd\nprint('hello world')\n"
        assert code_manager._clean_code(pandas_code, exec_context) == "print('hello world')"

    def test_clean_code_whitelist_import(self, code_manager: CodeManager, exec_context: MagicMock):
        if False:
            while True:
                i = 10
        'Test that an installed whitelisted library is added to the environment.'
        safe_code = '\nimport numpy as np\nnp.array()\n'
        assert code_manager._clean_code(safe_code, exec_context) == 'np.array()'

    def test_clean_code_raise_bad_import_error(self, code_manager: CodeManager, exec_context: MagicMock):
        if False:
            i = 10
            return i + 15
        malicious_code = '\nimport os\nprint(os.listdir())\n'
        with pytest.raises(BadImportError):
            code_manager.execute_code(malicious_code, exec_context)

    def test_remove_dfs_overwrites(self, code_manager: CodeManager, exec_context: MagicMock):
        if False:
            return 10
        hallucinated_code = 'def analyze_data(dfs):\n    dfs = [pd.DataFrame([1,2,3])]\n    print(dfs)'
        assert code_manager._clean_code(hallucinated_code, exec_context) == 'def analyze_data(dfs):\n    print(dfs)'

    def test_exception_handling(self, smart_dataframe: SmartDataframe, code_manager: CodeManager):
        if False:
            print('Hello World!')
        code_manager.execute_code = Mock(side_effect=NoCodeFoundError('No code found in the answer.'))
        code_manager.execute_code.__name__ = 'execute_code'
        result = smart_dataframe.chat('How many countries are in the dataframe?')
        assert result == 'Unfortunately, I was not able to answer your question, because of the following error:\n\nNo code found in the answer.\n'
        assert smart_dataframe.last_error == 'No code found in the answer.'

    def test_custom_whitelisted_dependencies(self, code_manager: CodeManager, llm, exec_context: MagicMock):
        if False:
            print('Hello World!')
        code = '\nimport my_custom_library\ndef analyze_data(dfs: list):\n    my_custom_library.do_something()\n'
        llm._output = code
        with pytest.raises(BadImportError):
            code_manager._clean_code(code, exec_context)
        code_manager._config.custom_whitelisted_dependencies = ['my_custom_library']
        assert code_manager._clean_code(code, exec_context) == 'def analyze_data(dfs: list):\n    my_custom_library.do_something()'

    def test_get_environment(self, code_manager: CodeManager, smart_dataframe):
        if False:
            print('Hello World!')
        code_manager._additional_dependencies = [{'name': 'pyplot', 'alias': 'plt', 'module': 'matplotlib'}, {'name': 'numpy', 'alias': 'np', 'module': 'numpy'}]
        assert 'pd' in code_manager._get_environment()
        assert 'plt' in code_manager._get_environment()
        assert 'np' in code_manager._get_environment()
        assert code_manager._get_environment()['__builtins__'] == {'abs': abs, 'all': all, 'any': any, 'ascii': ascii, 'bin': bin, 'bool': bool, 'bytearray': bytearray, 'bytes': bytes, 'callable': callable, 'chr': chr, 'classmethod': classmethod, 'complex': complex, 'delattr': delattr, 'dict': dict, 'dir': dir, 'divmod': divmod, 'enumerate': enumerate, 'filter': filter, 'float': float, 'format': format, 'frozenset': frozenset, 'getattr': getattr, 'hasattr': hasattr, 'hash': hash, 'help': help, 'hex': hex, 'id': id, 'int': int, 'isinstance': isinstance, 'issubclass': issubclass, 'iter': iter, 'len': len, 'list': list, 'locals': locals, 'map': map, 'max': max, 'memoryview': memoryview, 'min': min, 'next': next, 'object': object, 'oct': oct, 'ord': ord, 'pow': pow, 'print': print, 'property': property, 'range': range, 'repr': repr, 'reversed': reversed, 'round': round, 'set': set, 'setattr': setattr, 'slice': slice, 'sorted': sorted, 'staticmethod': staticmethod, 'str': str, 'sum': sum, 'super': super, 'tuple': tuple, 'type': type, 'vars': vars, 'zip': zip, '__build_class__': __build_class__, '__name__': '__main__'}

    def test_execute_catching_errors_correct(self, code_manager: CodeManager):
        if False:
            while True:
                i = 10
        code = "def analyze_data(dfs):\n    return {'type': 'number', 'value': 1 + 1}"
        environment = {'dfs': []}
        with patch('builtins.exec') as mock_exec:
            assert code_manager._execute_catching_errors(code, environment) is None
            mock_exec.assert_called_once_with(code + '\n\nresult = analyze_data(dfs)', environment)

    def test_execute_catching_errors_raise_exc(self, code_manager: CodeManager):
        if False:
            i = 10
            return i + 15
        code = 'def analyze_data(dfs):\n    raise RuntimeError()'
        environment = {'dfs': []}
        with patch('builtins.exec') as mock_exec:
            mock_exec.side_effect = RuntimeError('foobar')
            exc = code_manager._execute_catching_errors(code, environment)
            mock_exec.assert_called_once_with(code + '\n\nresult = analyze_data(dfs)', environment)
            assert isinstance(exc, RuntimeError)

    def test_handle_error_name_error(self, code_manager: CodeManager):
        if False:
            return 10
        code = 'def analyze_data(dfs):\n    print(json.dumps({"foo": "bar"}))'
        environment = {'dfs': []}
        exc = code_manager._execute_catching_errors(code, environment)
        code_manager._handle_error(exc, code, environment)
        assert getattr(environment.get('json'), '__name__', None) == 'json'

    def test_handle_error_name_error_not_whitelisted_lib(self, code_manager: CodeManager):
        if False:
            print('Hello World!')
        code = 'def analyze_data(dfs):\n    print(os)'
        environment = {'dfs': []}
        exc = code_manager._execute_catching_errors(code, environment)
        with pytest.raises(NameError):
            code_manager._handle_error(exc, code, environment, use_error_correction_framework=False)
        assert 'os' not in environment

    @pytest.mark.parametrize('df_name, code', [('df', "\ndef analyze_data(dfs: list[pd.DataFrame]) -> dict:\n    df = dfs[0]\n    filtered_df = df.filter(\n        (pl.col('loan_status') == 'PAIDOFF') & (pl.col('Gender') == 'male')\n    )\n    count = filtered_df.shape[0]\n    result = {'type': 'number', 'value': count}\n    return result\n\nresult = analyze_data(dfs)\n                "), ('foobar', "\ndef analyze_data(dfs: list[pd.DataFrame]) -> dict:\n    foobar = dfs[0]\n    filtered_df = foobar.filter(\n        (pl.col('loan_status') == 'PAIDOFF') & (pl.col('Gender') == 'male')\n    )\n    count = filtered_df.shape[0]\n    result = {'type': 'number', 'value': count}\n    return result\n\nresult = analyze_data(dfs)\n                ")])
    def test_extract_filters_polars(self, df_name, code, code_manager: CodeManager):
        if False:
            while True:
                i = 10
        filters = code_manager._extract_filters(code)
        assert isinstance(filters, dict)
        assert 'dfs[0]' in filters
        assert isinstance(filters['dfs[0]'], list)
        assert len(filters['dfs[0]']) == 2
        assert filters['dfs[0]'][0] == ('loan_status', '=', 'PAIDOFF')
        assert filters['dfs[0]'][1] == ('Gender', '=', 'male')

    def test_extract_filters_polars_multiple_df(self, code_manager: CodeManager):
        if False:
            return 10
        code = "\ndef analyze_data(dfs: list[pd.DataFrame]) -> dict:\n    df = dfs[0]\n    filtered_paid_df_male = df.filter(\n        (pl.col('loan_status') == 'PAIDOFF') & (pl.col('Gender') == 'male')\n    )\n    num_loans_paid_off_male = len(filtered_paid_df)\n\n    df = dfs[1]\n    filtered_pend_df_male = df.filter(\n        (pl.col('loan_status') == 'PENDING') & (pl.col('Gender') == 'male')\n    )\n    num_loans_pending_male = len(filtered_pend_df)\n\n    df = dfs[2]\n    filtered_paid_df_female = df.filter(\n        (pl.col('loan_status') == 'PAIDOFF') & (pl.col('Gender') == 'female')\n    )\n    num_loans_paid_off_female = len(filtered_pend_df)\n\n    value = num_loans_paid_off + num_loans_pending + num_loans_paid_off_female\n    result = {\n        'type': 'number',\n        'value': value\n    }\n    return result\n\nresult = analyze_data(dfs)\n"
        filters = code_manager._extract_filters(code)
        assert isinstance(filters, dict)
        assert 'dfs[0]' in filters
        assert 'dfs[1]' in filters
        assert 'dfs[2]' in filters
        assert isinstance(filters['dfs[0]'], list)
        assert len(filters['dfs[0]']) == 2
        assert len(filters['dfs[1]']) == 2
        assert filters['dfs[0]'][0] == ('loan_status', '=', 'PAIDOFF')
        assert filters['dfs[0]'][1] == ('Gender', '=', 'male')
        assert filters['dfs[1]'][0] == ('loan_status', '=', 'PENDING')
        assert filters['dfs[1]'][1] == ('Gender', '=', 'male')
        assert filters['dfs[2]'][0] == ('loan_status', '=', 'PAIDOFF')
        assert filters['dfs[2]'][1] == ('Gender', '=', 'female')

    @pytest.mark.parametrize('df_name', ['df', 'foobar'])
    def test_extract_filters_col_index(self, df_name, code_manager: CodeManager):
        if False:
            i = 10
            return i + 15
        code = f"\ndef analyze_data(dfs: list[pd.DataFrame]) -> dict:\n    {df_name} = dfs[0]\n    filtered_df = (\n        {df_name}[\n            ({df_name}['loan_status'] == 'PAIDOFF') & ({df_name}['Gender'] == 'male')\n        ]\n    )\n    num_loans = len(filtered_df)\n    result = {{'type': 'number', 'value': num_loans}}\n    return result\n\nresult = analyze_data(dfs)\n"
        filters = code_manager._extract_filters(code)
        assert isinstance(filters, dict)
        assert 'dfs[0]' in filters
        assert isinstance(filters['dfs[0]'], list)
        assert len(filters['dfs[0]']) == 2
        assert filters['dfs[0]'][0] == ('loan_status', '=', 'PAIDOFF')
        assert filters['dfs[0]'][1] == ('Gender', '=', 'male')

    @pytest.mark.parametrize('df_name, code', [('df', "\ndef analyze_data(dfs: list[pd.DataFrame]) -> dict:\n    df = dfs[0]\n    filtered_df = df.filter(\n        (pl.col('loan_status') == 'PAIDOFF') & (pl.col('Gender') == 'male')\n    )\n    count = filtered_df.shape[0]\n    result = {'type': 'number', 'value': count}\n    return result\n\nresult = analyze_data(dfs)\n                "), ('foobar', "\ndef analyze_data(dfs: list[pd.DataFrame]) -> dict:\n    foobar = dfs[0]\n    filtered_df = foobar[(\n        foobar['loan_status'] == 'PAIDOFF'\n    ) & (df['Gender'] == 'male')]\n    num_loans = len(filtered_df)\n    result = {'type': 'number', 'value': num_loans}\n    return result\n\nresult = analyze_data(dfs)\n                ")])
    def test_extract_filters_col_index_non_default_name(self, df_name, code, code_manager: CodeManager):
        if False:
            i = 10
            return i + 15
        filters = code_manager._extract_filters(code)
        assert isinstance(filters, dict)
        assert 'dfs[0]' in filters
        assert isinstance(filters['dfs[0]'], list)
        assert len(filters['dfs[0]']) == 2
        assert filters['dfs[0]'][0] == ('loan_status', '=', 'PAIDOFF')
        assert filters['dfs[0]'][1] == ('Gender', '=', 'male')

    def test_extract_filters_col_index_multiple_df(self, code_manager: CodeManager):
        if False:
            while True:
                i = 10
        code = "\ndef analyze_data(dfs: list[pd.DataFrame]) -> dict:\n    df = dfs[0]\n    filtered_paid_df_male = df[(\n        df['loan_status'] == 'PAIDOFF') & (df['Gender'] == 'male'\n    )]\n    num_loans_paid_off_male = len(filtered_paid_df)\n\n    df = dfs[1]\n    filtered_pend_df_male = df[(\n        df['loan_status'] == 'PENDING') & (df['Gender'] == 'male'\n    )]\n    num_loans_pending_male = len(filtered_pend_df)\n\n    df = dfs[2]\n    filtered_paid_df_female = df[(\n        df['loan_status'] == 'PAIDOFF') & (df['Gender'] == 'female'\n    )]\n    num_loans_paid_off_female = len(filtered_pend_df)\n\n    value = num_loans_paid_off + num_loans_pending + num_loans_paid_off_female\n    result = {\n        'type': 'number',\n        'value': value\n    }\n    return result\n\nresult = analyze_data(dfs)\n"
        filters = code_manager._extract_filters(code)
        assert isinstance(filters, dict)
        assert 'dfs[0]' in filters
        assert 'dfs[1]' in filters
        assert 'dfs[2]' in filters
        assert isinstance(filters['dfs[0]'], list)
        assert len(filters['dfs[0]']) == 2
        assert len(filters['dfs[1]']) == 2
        assert filters['dfs[0]'][0] == ('loan_status', '=', 'PAIDOFF')
        assert filters['dfs[0]'][1] == ('Gender', '=', 'male')
        assert filters['dfs[1]'][0] == ('loan_status', '=', 'PENDING')
        assert filters['dfs[1]'][1] == ('Gender', '=', 'male')
        assert filters['dfs[2]'][0] == ('loan_status', '=', 'PAIDOFF')
        assert filters['dfs[2]'][1] == ('Gender', '=', 'female')