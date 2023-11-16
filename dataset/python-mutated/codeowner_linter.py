"""Lint checks for codeowner file."""
from __future__ import annotations
import glob
import os
import subprocess
from typing import Final, Iterator, List, Tuple
from . import linter_utils
from .. import concurrent_task_utils
MYPY = False
if MYPY:
    from scripts.linters import pre_commit_linter
CODEOWNER_FILEPATH: Final = '.github/CODEOWNERS'
CODEOWNER_IMPORTANT_PATHS: Final = ['/core/storage/', '/dependencies.json', '/package.json', '/requirements.txt', '/requirements.in', '/requirements_dev.txt', '/requirements_dev.in', '/yarn.lock', '/scripts/install_third_party_libs.py', '/.github/', '/.github/CODEOWNERS', '/.github/stale.yml', '/.github/workflows/', '/core/android_validation_constants*.py', '/extensions/interactions/rule_templates.json', '/core/templates/services/svg-sanitizer.service.ts', '/scripts/linters/warranted_angular_security_bypasses.py', '/core/controllers/access_validators*.py', '/core/controllers/acl_decorators*.py', '/core/controllers/android*.py', '/core/controllers/base*.py', '/core/controllers/firebase*.py', '/core/domain/android*.py', '/core/domain/html*.py', '/core/domain/rights_manager*.py', '/core/domain/role_services*.py', '/core/domain/user*.py', '/AUTHORS', '/CONTRIBUTORS', '/LICENSE', '/NOTICE', '/core/templates/pages/terms-page/terms-page.component.html', '/core/templates/pages/privacy-page/privacy-page.component.html', '/core/templates/pages/license-page/license-page.component.html', '/core/domain/takeout_*.py', '/core/domain/wipeout_*.py']

class CodeownerLintChecksManager(linter_utils.BaseLinter):
    """Manages codeowner checks."""

    def __init__(self, file_cache: pre_commit_linter.FileCache) -> None:
        if False:
            i = 10
            return i + 15
        'Constructs a CodeownerLintChecksManager object.\n\n        Args:\n            file_cache: object(FileCache). Provides thread-safe access to cached\n                file content.\n        '
        self.file_cache = file_cache
        self.error_messages: List[str] = []
        self.failed = False

    def _walk_with_gitignore(self, root: str, exclude_dirs: List[str]) -> Iterator[List[str]]:
        if False:
            for i in range(10):
                print('nop')
        'A walk function similar to os.walk but this would ignore the files\n        and directories which is not tracked by git. Also, this will ignore the\n        directories mentioned in exclude_dirs.\n\n        Args:\n            root: str. The path from where the function should start walking.\n            exclude_dirs: list(str). A list of dir path which should be ignored.\n\n        Yields:\n            list(str). A list of unignored files.\n        '
        (dirs, file_paths) = ([], [])
        for name in os.listdir(root):
            if os.path.isdir(os.path.join(root, name)):
                dirs.append(os.path.join(root, name))
            else:
                file_paths.append(os.path.join(root, name))
        yield [file_path for file_path in file_paths if not self._is_path_ignored(file_path)]
        for dir_path in dirs:
            if not self._is_path_ignored(dir_path + '/') and dir_path not in exclude_dirs:
                for x in self._walk_with_gitignore(dir_path, exclude_dirs):
                    yield x

    def _is_path_ignored(self, path_to_check: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Checks whether the given path is ignored by git.\n\n        Args:\n            path_to_check: str. A path to a file or a dir.\n\n        Returns:\n            bool. Whether the given path is ignored by git.\n        '
        command = ['git', 'check-ignore', '-q', path_to_check]
        return subprocess.call(command) == 0

    def _is_path_contains_frontend_specs(self, path_to_check: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Checks whether if a path contains all spec files.\n\n        Args:\n            path_to_check: str. A path to a file or a dir.\n\n        Returns:\n            bool. Whether the given path contains all spec files.\n        '
        return '*.spec.ts' in path_to_check or '*Spec.ts' in path_to_check

    def _check_for_important_patterns_at_bottom_of_codeowners(self, important_patterns: List[str]) -> None:
        if False:
            print('Hello World!')
        'Checks that the most important patterns are at the bottom\n        of the CODEOWNERS file.\n\n        Args:\n            important_patterns: list(str). List of the important\n                patterns for CODEOWNERS file.\n        '
        important_patterns_set = set(important_patterns)
        codeowner_important_paths_set = set(CODEOWNER_IMPORTANT_PATHS)
        if len(important_patterns_set) != len(important_patterns):
            error_message = '%s --> Duplicate pattern(s) found in critical rules section.' % CODEOWNER_FILEPATH
            self.error_messages.append(error_message)
            self.failed = True
        if len(codeowner_important_paths_set) != len(CODEOWNER_IMPORTANT_PATHS):
            error_message = 'scripts/linters/codeowner_linter.py --> Duplicate pattern(s) found in CODEOWNER_IMPORTANT_PATHS list.'
            self.error_messages.append(error_message)
            self.failed = True
        critical_rule_section_minus_list_set = important_patterns_set.difference(codeowner_important_paths_set)
        list_minus_critical_rule_section_set = codeowner_important_paths_set.difference(important_patterns_set)
        for rule in critical_rule_section_minus_list_set:
            error_message = "%s --> Rule %s is not present in the CODEOWNER_IMPORTANT_PATHS list in scripts/linters/codeowner_linter.py. Please add this rule in the mentioned list or remove this rule from the 'Critical files' section." % (CODEOWNER_FILEPATH, rule)
            self.error_messages.append(error_message)
            self.failed = True
        for rule in list_minus_critical_rule_section_set:
            error_message = "%s --> Rule '%s' is not present in the 'Critical files' section. Please place it under the 'Critical files' section since it is an important rule. Alternatively please remove it from the 'CODEOWNER_IMPORTANT_PATHS' list in scripts/linters/codeowner_linter.py if it is no longer an important rule." % (CODEOWNER_FILEPATH, rule)
            self.error_messages.append(error_message)
            self.failed = True

    def check_codeowner_file(self) -> concurrent_task_utils.TaskResult:
        if False:
            for i in range(10):
                print('nop')
        'Checks the CODEOWNERS file for any uncovered dirs/files and also\n        checks that every pattern in the CODEOWNERS file matches at least one\n        file/dir. Note that this checks the CODEOWNERS file according to the\n        glob patterns supported by Python2.7 environment. For more information\n        please refer https://docs.python.org/2/library/glob.html.\n        This function also ensures that the most important rules are at the\n        bottom of the CODEOWNERS file.\n\n        Returns:\n            TaskResult. A TaskResult object representing the result of the lint\n            check.\n        '
        name = 'CODEOWNERS'
        critical_file_section_found = False
        inside_blanket_codeowners_section = False
        important_rules_in_critical_section = []
        file_patterns = []
        ignored_dir_patterns = []
        for (line_num, line) in enumerate(self.file_cache.readlines(CODEOWNER_FILEPATH)):
            stripped_line = line.strip()
            if '# Critical files' in line:
                critical_file_section_found = True
            if '# Blanket codeowners' in line:
                inside_blanket_codeowners_section = True
            if inside_blanket_codeowners_section is True and (not stripped_line):
                inside_blanket_codeowners_section = False
                continue
            if stripped_line and stripped_line[0] != '#':
                if '#' in line:
                    error_message = '%s --> Please remove inline comment from line %s' % (CODEOWNER_FILEPATH, line_num + 1)
                    self.error_messages.append(error_message)
                    self.failed = True
                if '@' not in line:
                    error_message = "%s --> Pattern on line %s doesn't have codeowner" % (CODEOWNER_FILEPATH, line_num + 1)
                    self.error_messages.append(error_message)
                    self.failed = True
                else:
                    line_in_concern = line.split('@')[0].strip()
                    if critical_file_section_found:
                        important_rules_in_critical_section.append(line_in_concern)
                    if not line_in_concern.startswith('/'):
                        error_message = '%s --> Pattern on line %s is invalid. Use full path relative to the root directory' % (CODEOWNER_FILEPATH, line_num + 1)
                        self.error_messages.append(error_message)
                        self.failed = True
                    if not self._is_path_contains_frontend_specs(line_in_concern):
                        if '**' in line_in_concern:
                            error_message = "%s --> Pattern on line %s is invalid. '**' wildcard not allowed" % (CODEOWNER_FILEPATH, line_num + 1)
                            self.error_messages.append(error_message)
                            self.failed = True
                    if line_in_concern.endswith('/'):
                        line_in_concern = line_in_concern[:-1]
                    line_in_concern = line_in_concern.replace('/', './', 1)
                    if not self._is_path_contains_frontend_specs(line_in_concern):
                        if not glob.glob(line_in_concern):
                            error_message = "%s --> Pattern on line %s doesn't match any file or directory" % (CODEOWNER_FILEPATH, line_num + 1)
                            self.error_messages.append(error_message)
                            self.failed = True
                    if not inside_blanket_codeowners_section:
                        if os.path.isdir(line_in_concern):
                            ignored_dir_patterns.append(line_in_concern)
                        else:
                            file_patterns.append(line_in_concern)
        for file_paths in self._walk_with_gitignore('.', ignored_dir_patterns):
            for file_path in file_paths:
                match = False
                for file_pattern in file_patterns:
                    if file_path in glob.glob(file_pattern):
                        match = True
                        break
                if not match:
                    error_message = '%s is not listed in the .github/CODEOWNERS file.' % file_path
                    self.error_messages.append(error_message)
                    self.failed = True
        self._check_for_important_patterns_at_bottom_of_codeowners(important_rules_in_critical_section)
        return concurrent_task_utils.TaskResult(name, self.failed, self.error_messages, self.error_messages)

    def perform_all_lint_checks(self) -> List[concurrent_task_utils.TaskResult]:
        if False:
            while True:
                i = 10
        'Perform all the lint checks and returns the messages returned by all\n        the checks.\n\n        Returns:\n            list(TaskResult). A list of TaskResult objects representing the\n            results of the lint checks.\n        '
        return [self.check_codeowner_file()]

def get_linters(file_cache: pre_commit_linter.FileCache) -> Tuple[CodeownerLintChecksManager, None]:
    if False:
        while True:
            i = 10
    'Creates CodeownerLintChecksManager object and returns it.\n\n    Args:\n        file_cache: object(FileCache). Provides thread-safe access to cached\n            file content.\n\n    Returns:\n        tuple(CodeownerLintChecksManager, None). A 2-tuple of custom and\n        third_party linter objects.\n    '
    custom_linter = CodeownerLintChecksManager(file_cache)
    return (custom_linter, None)