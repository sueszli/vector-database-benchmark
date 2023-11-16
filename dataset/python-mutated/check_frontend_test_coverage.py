"""Check for decrease in coverage from 100% of frontend files."""
from __future__ import annotations
import fnmatch
import logging
import os
import re
import sys
from core import utils
from typing import List
LCOV_FILE_PATH = os.path.join(os.pardir, 'karma_coverage_reports', 'lcov.info')
RELEVANT_LCOV_LINE_PREFIXES = ['SF', 'LH', 'LF']
EXCLUDED_DIRECTORIES = ['node_modules/*', 'extensions/classifiers/proto/*']
NOT_FULLY_COVERED_FILENAMES = ['angular-html-bind.directive.ts', 'App.ts', 'Base.ts', 'ck-editor-4-rte.component.ts', 'ck-editor-4-widgets.initializer.ts', 'exploration-states.service.ts', 'expression-interpolation.service.ts', 'google-analytics.initializer.ts', 'learner-answer-info.service.ts', 'mathjax-bind.directive.ts', 'object-editor.directive.ts', 'oppia-interactive-music-notes-input.component.ts', 'oppia-interactive-pencil-code-editor.component.ts', 'oppia-root.directive.ts', 'python-program.tokenizer.ts', 'question-update.service.ts', 'questions-list-select-skill-and-difficulty-modal.component.ts', 'questions-opportunities-select-difficulty-modal.component.ts', 'rte-helper-modal.controller.ts', 'rule-type-selector.directive.ts', 'translation-file-hash-loader-backend-api.service.ts', 'unit-test-utils.ajs.ts', 'voiceover-recording.service.ts']

class LcovStanzaRelevantLines:
    """Gets the relevant lines from a lcov stanza."""

    def __init__(self, stanza: str) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize the object which provides relevant data of a lcov\n        stanza in order to calculate any decrease in frontend test coverage.\n\n        Args:\n            stanza: list(str). Contains all the lines from a lcov stanza.\n\n        Raises:\n            Exception. The file_path is empty.\n            Exception. Total lines number is not found.\n            Exception. Covered lines number is not found.\n        '
        match = re.search('SF:(.+)\n', stanza)
        if match is None:
            raise Exception("The test path is empty or null. It's not possible to diff the test coverage correctly.")
        (_, file_name) = os.path.split(match.group(1))
        self.file_name = file_name
        self.file_path = match.group(1)
        match = re.search('LF:(\\d+)\\n', stanza)
        if match is None:
            raise Exception("It wasn't possible to get the total lines of {} file.It's not possible to diff the test coverage correctly.".format(file_name))
        self.total_lines = int(match.group(1))
        match = re.search('LH:(\\d+)\\n', stanza)
        if match is None:
            raise Exception("It wasn't possible to get the covered lines of {} file.It's not possible to diff the test coverage correctly.".format(file_name))
        self.covered_lines = int(match.group(1))

def get_stanzas_from_lcov_file() -> List[LcovStanzaRelevantLines]:
    if False:
        print('Hello World!')
    'Get all stanzas from a lcov file. The lcov file gather all the frontend\n    files that has tests and each one has the following structure:\n    TN: test name\n    SF: file path\n    FNF: total functions\n    FNH: functions covered\n    LF: total lines\n    LH: lines covered\n    BRF: total branches\n    BRH: branches covered\n    end_of_record\n\n    Returns:\n        list(LcovStanzaRelevantLines). A list with all stanzas.\n    '
    f = utils.open_file(LCOV_FILE_PATH, 'r')
    lcov_items_list = f.read().split('end_of_record')
    stanzas_list = []
    for item in lcov_items_list:
        if item.strip('\n'):
            stanza = LcovStanzaRelevantLines(item)
            stanzas_list.append(stanza)
    return stanzas_list

def check_not_fully_covered_filenames_list_is_sorted() -> None:
    if False:
        i = 10
        return i + 15
    'Check if NOT_FULLY_COVERED_FILENAMES list is in alphabetical order.'
    if NOT_FULLY_COVERED_FILENAMES != sorted(NOT_FULLY_COVERED_FILENAMES, key=lambda s: s.lower()):
        logging.error('The \x1b[1mNOT_FULLY_COVERED_FILENAMES\x1b[0m list must be kept in alphabetical order.')
        sys.exit(1)

def check_coverage_changes() -> None:
    if False:
        return 10
    "Checks if the denylist for not fully covered files needs to be changed\n    by:\n    - File renaming\n    - File deletion\n\n    Raises:\n        Exception. LCOV_FILE_PATH doesn't exist.\n    "
    if not os.path.exists(LCOV_FILE_PATH):
        raise Exception('Expected lcov file to be available at {}, but the file does not exist.'.format(LCOV_FILE_PATH))
    stanzas = get_stanzas_from_lcov_file()
    remaining_denylisted_files = list(NOT_FULLY_COVERED_FILENAMES)
    errors = ''
    for stanza in stanzas:
        file_name = stanza.file_name
        total_lines = stanza.total_lines
        covered_lines = stanza.covered_lines
        if any((fnmatch.fnmatch(stanza.file_path, pattern) for pattern in EXCLUDED_DIRECTORIES)):
            continue
        if file_name not in remaining_denylisted_files:
            if total_lines != covered_lines:
                errors += "\x1b[1m{}\x1b[0m seems to be not completely tested. Make sure it's fully covered.\n".format(file_name)
        else:
            if total_lines == covered_lines:
                errors += "\x1b[1m{}\x1b[0m seems to be fully covered! Before removing it manually from the denylist in the file scripts/check_frontend_test_coverage.py, please make sure you've followed the unit tests rules correctly on: https://github.com/oppia/oppia/wiki/Frontend-unit-tests-guide#rules\n".format(file_name)
            remaining_denylisted_files.remove(file_name)
    if remaining_denylisted_files:
        for test_name in remaining_denylisted_files:
            errors += "\x1b[1m{}\x1b[0m is in the frontend test coverage denylist but it doesn't exist anymore. If you have renamed it, please make sure to remove the old file name and add the new file name in the denylist in the file scripts/check_frontend_test_coverage.py.\n".format(test_name)
    if errors:
        print('------------------------------------')
        print('Frontend Coverage Checks Not Passed.')
        print('------------------------------------')
        logging.error(errors)
        sys.exit(1)
    else:
        print('------------------------------------')
        print('All Frontend Coverage Checks Passed.')
        print('------------------------------------')
    check_not_fully_covered_filenames_list_is_sorted()

def main() -> None:
    if False:
        while True:
            i = 10
    'Runs all the steps for checking if there is any decrease of 100% covered\n    files in the frontend.\n    '
    check_coverage_changes()
if __name__ == '__main__':
    main()