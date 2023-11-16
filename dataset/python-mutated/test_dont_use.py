import os
import re
import codecs
import pytest
root_path = os.path.realpath(os.path.join(os.getcwd(), 'spyder'))

@pytest.mark.parametrize('pattern,exclude_patterns,message', [('isinstance\\(.*,.*str\\)', ['py3compat.py', 'example_latin1.py'], "Don't use builtin isinstance() function,use spyder.py3compat.is_text_string() instead"), ('^[\\s\\#]*\\bprint\\(((?!file=).)*\\)', ['.*test.*', 'example.py', 'example_latin1.py', 'binaryornot'], ("Don't use the print() function; ", 'for debugging, use logging module instead')), ('^[\\s\\#]*\\bprint\\s+(?!>>)((?!#).)*', ['.*test.*', 'example_latin1.py'], ("Don't use print statements; ", 'for debugging, use the logging module instead.'))])
def test_dont_use(pattern, exclude_patterns, message):
    if False:
        while True:
            i = 10
    '\n    This test is used for discouraged using of some expressions that could\n    introduce errors, and encourage use spyder function instead.\n\n    If you want to skip some line from this test just use:\n        # spyder: test-skip\n    '
    pattern = re.compile(pattern + '((?!# spyder: test-skip)\\s)*$')
    found = 0
    for (dir_name, _, file_list) in os.walk(root_path):
        for fname in file_list:
            exclude = any([re.search(ex, fname) for ex in exclude_patterns])
            exclude = exclude or any([re.search(ex, dir_name) for ex in exclude_patterns])
            if fname.endswith('.py') and (not exclude):
                file = os.path.join(dir_name, fname)
                with codecs.open(file, encoding='utf-8') as f:
                    for (i, line) in enumerate(f):
                        for match in re.finditer(pattern, line):
                            print('{}\nline:{}, {}'.format(file, i + 1, line))
                            found += 1
    assert found == 0, '{}\n{} errors found'.format(message, found)

@pytest.mark.parametrize('pattern', [u'％'])
def test_check_charaters_translation(pattern):
    if False:
        i = 10
        return i + 15
    u'\n    This test is used to prevent the addition of unwanted unicode characters\n    in the translations like ％ instead of %.\n\n    '
    found = 0
    for (dir_name, _, file_list) in os.walk(os.path.join(root_path, 'locale')):
        for fname in file_list:
            if fname.endswith('.po'):
                file = os.path.join(dir_name, fname)
                with codecs.open(file, encoding='utf-8') as f:
                    for (i, line) in enumerate(f):
                        for match in re.finditer(pattern, line):
                            print(u'{}\nline:{}, {}'.format(file, i + 1, line))
                            found += 1
    assert found == 0, u'{}\n{} characters found'.format(u'Strange characters found in translations', found)