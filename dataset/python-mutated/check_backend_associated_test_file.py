"""This script checks if every backend file has an associated test file."""
from __future__ import annotations
import logging
import os
import subprocess
import sys
TOPMOST_LEVEL_PATH = './'
FILES_WITHOUT_ASSOCIATED_TEST_FILES = ['scripts/linters/test_files/invalid_annotations.py', 'scripts/linters/test_files/invalid_author.py', 'scripts/linters/test_files/invalid_copyright.py', 'scripts/linters/test_files/invalid_docstring.py', 'scripts/linters/test_files/invalid_import_order.py', 'scripts/linters/test_files/invalid_merge_conflict.py', 'scripts/linters/test_files/invalid_datastore.py', 'scripts/linters/test_files/invalid_no_newline.py', 'scripts/linters/test_files/invalid_pycodestyle_error.py', 'scripts/linters/test_files/invalid_pylint_id.py', 'scripts/linters/test_files/invalid_python_three.py', 'scripts/linters/test_files/invalid_request.py', 'scripts/linters/test_files/invalid_tabs.py', 'scripts/linters/test_files/invalid_todo.py', 'scripts/linters/test_files/invalid_urlopen.py', 'scripts/linters/test_files/valid.py', 'scripts/linters/test_files/valid_job_imports.py', 'scripts/linters/test_files/valid_py_ignore_pragma.py', 'core/tests/build_sources/extensions/CodeRepl.py', 'core/tests/build_sources/extensions/DragAndDropSortInput.py', 'core/tests/data/failing_tests.py', 'core/tests/data/image_constants.py', 'core/tests/data/unicode_and_str_handler.py', 'proto_files/text_classifier_pb2.py', 'proto_files/training_job_response_payload_pb2.py']

def check_if_path_ignored(path_to_check: str) -> bool:
    if False:
        print('Hello World!')
    'Checks whether the given path is ignored by git.\n\n    Args:\n        path_to_check: str. A path to a file or a dir.\n\n    Returns:\n        bool. Whether the given path is ignored by git.\n    '
    command = ['git', 'check-ignore', '-q', path_to_check]
    return subprocess.call(command) == 0

def main() -> None:
    if False:
        i = 10
        return i + 15
    'Finds the non-empty backend files that lack an associated test file.'
    all_backend_files = []
    for (root, _, files) in os.walk(TOPMOST_LEVEL_PATH):
        for file in files:
            full_path = os.path.join(root, file)
            (_, file_extension) = os.path.splitext(full_path)
            if file_extension == '.py' and (not check_if_path_ignored(full_path)):
                all_backend_files.append(full_path)
    all_backend_files.sort()
    files_without_test = []
    for file in all_backend_files:
        if not file.endswith('_test.py'):
            test_file_name = '%s_test.py' % file[:-3]
            if test_file_name not in all_backend_files:
                files_without_test.append(file)
    non_empty_files = []
    for file in files_without_test:
        line_count = 0
        with open(file, 'r', encoding='utf8') as f:
            line_count = len(f.readlines())
        if line_count > 0:
            non_empty_files.append(file[2:])
    errors = ''
    for file in non_empty_files:
        if file not in FILES_WITHOUT_ASSOCIATED_TEST_FILES:
            errors += '\x1b[1m{}\x1b[0m needs an associated backend test file.\n'.format(file)
    if errors:
        print('-------------------------------------------')
        print('Backend associated test file checks failed.')
        print('-------------------------------------------')
        logging.error(errors)
        sys.exit(1)
    else:
        print('-------------------------------------------')
        print('Backend associated test file checks passed.')
        print('-------------------------------------------')
if __name__ == '__main__':
    main()