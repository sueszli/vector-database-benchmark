import pytest
from numpydoc.validate import Docstring
from scripts.doc_checker import MODIN_ERROR_CODES, check_optional_args, check_spelling_words, get_noqa_checks, get_optional_args

@pytest.mark.parametrize('import_path, result', [('scripts.test.examples.optional_square', {'number': 5}), ('scripts.test.examples.optional_square_empty_parameters', {'number': 5}), ('scripts.test.examples.square_summary', {}), ('scripts.test.examples.weakdict', {}), ('scripts.test.examples', {})])
def test_get_optional_args(import_path, result):
    if False:
        return 10
    optional_args = get_optional_args(Docstring(import_path))
    assert optional_args == result

@pytest.mark.parametrize('import_path, result', [('scripts.test.examples.optional_square', [('MD01', MODIN_ERROR_CODES['MD01'].format(parameter='number', found='int'))]), ('scripts.test.examples.optional_square_empty_parameters', []), ('scripts.test.examples.square_summary', []), ('scripts.test.examples.weakdict', []), ('scripts.test.examples', [])])
def test_check_optional_args(import_path, result):
    if False:
        for i in range(10):
            print('nop')
    errors = check_optional_args(Docstring(import_path))
    assert errors == result

@pytest.mark.parametrize('import_path, result', [('scripts.test.examples.optional_square', []), ('scripts.test.examples.square_summary', [('MD02', 57, 'Pandas', 'pandas'), ('MD02', 57, 'Numpy', 'NumPy')]), ('scripts.test.examples.optional_square_empty_parameters', []), ('scripts.test.examples.weakdict', []), ('scripts.test.examples', [])])
def test_check_spelling_words(import_path, result):
    if False:
        while True:
            i = 10
    result_errors = []
    for (code, line, word, reference) in result:
        result_errors.append((code, MODIN_ERROR_CODES[code].format(line=line, word=word, reference=reference)))
    errors = check_spelling_words(Docstring(import_path))
    for error in errors:
        assert error in result_errors

@pytest.mark.parametrize('import_path, result', [('scripts.test.examples.optional_square', ['all']), ('scripts.test.examples.optional_square_empty_parameters', []), ('scripts.test.examples.square_summary', ['PR01', 'GL08']), ('scripts.test.examples.weakdict', ['GL08']), ('scripts.test.examples', ['MD02'])])
def test_get_noqa_checks(import_path, result):
    if False:
        i = 10
        return i + 15
    noqa_checks = get_noqa_checks(Docstring(import_path))
    assert noqa_checks == result