"""Lint checks for HTML files."""
from __future__ import annotations
import html.parser
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple
from . import linter_utils
from .. import common
from .. import concurrent_task_utils
MYPY = False
if MYPY:
    from scripts.linters import pre_commit_linter

class TagMismatchException(Exception):
    """Error class for mismatch between start and end tags."""
    pass

class CustomHTMLParser(html.parser.HTMLParser):
    """Custom HTML parser to check indentation."""

    def __init__(self, filepath: str, file_lines: Tuple[str, ...], failed: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Define various variables to parse HTML.\n\n        Args:\n            filepath: str. Path of the file.\n            file_lines: tuple(str). List of the lines in the file.\n            failed: bool. True if the HTML indentation check fails.\n        '
        html.parser.HTMLParser.__init__(self)
        self.error_messages: List[str] = []
        self.tag_stack: List[Tuple[str, int, int]] = []
        self.failed = failed
        self.filepath = filepath
        self.file_lines = file_lines
        self.indentation_level = 0
        self.indentation_width = 2
        self.void_elements = ['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr']

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if False:
            return 10
        'Handle start tag of a HTML line.\n\n        Args:\n            tag: str. Start tag of a HTML line.\n            attrs: List(Tuple[str, Optional[str]]). List of attributes\n                in the start tag.\n        '
        (line_number, column_number) = self.getpos()
        expected_indentation = self.indentation_level * self.indentation_width
        tag_line = self.file_lines[line_number - 1].lstrip()
        opening_tag = '<' + tag
        attr_pos_mapping: Dict[str, List[int]] = {}
        if tag_line.startswith(opening_tag) and tag == 'style':
            next_line = self.file_lines[line_number]
            next_line_expected_indentation = (self.indentation_level + 1) * self.indentation_width
            next_line_column_number = len(next_line) - len(next_line.lstrip())
            if next_line_column_number != next_line_expected_indentation:
                error_message = '%s --> Expected indentation of %s, found indentation of %s for content of %s tag on line %s ' % (self.filepath, next_line_expected_indentation, next_line_column_number, tag, line_number + 1)
                self.error_messages.append(error_message)
                self.failed = True
        if tag_line.startswith(opening_tag) and column_number != expected_indentation:
            error_message = '%s --> Expected indentation of %s, found indentation of %s for %s tag on line %s ' % (self.filepath, expected_indentation, column_number, tag, line_number)
            self.error_messages.append(error_message)
            self.failed = True
        if tag not in self.void_elements:
            self.tag_stack.append((tag, line_number, column_number))
            self.indentation_level += 1
        indentation_of_first_attribute = column_number + len(tag) + 2
        starttag_text = self.get_starttag_text()
        assert starttag_text is not None
        for (attr, value) in attrs:
            if value:
                value_in_quotes = True
                if '&quot;' in starttag_text:
                    expected_value = value
                    rendered_text = starttag_text.replace('&quot;', '"')
                else:
                    expected_value = '"' + value + '"'
                    rendered_text = starttag_text
                if not expected_value in rendered_text:
                    value_in_quotes = False
                    self.failed = True
                    error_message = '%s --> The value %s of attribute %s for the tag %s on line %s should be enclosed within double quotes.' % (self.filepath, value, attr, tag, line_number)
                    self.error_messages.append(error_message)
                self._check_space_between_attributes_and_values(tag, attr, value, rendered_text, value_in_quotes, attr_pos_mapping)
        for (line_num, line) in enumerate(starttag_text.splitlines()):
            if line_num == 0:
                continue
            leading_spaces_count = len(line) - len(line.lstrip())
            list_of_attrs = []
            for (attr, _) in attrs:
                list_of_attrs.append(attr)
            if not line.lstrip().startswith(tuple(list_of_attrs)):
                continue
            if indentation_of_first_attribute != leading_spaces_count:
                line_num_of_error = line_number + line_num
                error_message = '%s --> Attribute for tag %s on line %s should align with the leftmost attribute on line %s ' % (self.filepath, tag, line_num_of_error, line_number)
                self.error_messages.append(error_message)
                self.failed = True

    def handle_endtag(self, tag: str) -> None:
        if False:
            return 10
        'Handle end tag of a HTML line.\n\n        Args:\n            tag: str. End tag of a HTML line.\n\n        Raises:\n            TagMismatchException. Identation mismatch between starting tag and\n                given tag.\n        '
        (line_number, _) = self.getpos()
        tag_line = self.file_lines[line_number - 1]
        leading_spaces_count = len(tag_line) - len(tag_line.lstrip())
        try:
            (last_starttag, last_starttag_line_num, last_starttag_col_num) = self.tag_stack.pop()
        except IndexError as e:
            raise TagMismatchException('Error in line %s of file %s\n' % (line_number, self.filepath)) from e
        if last_starttag != tag:
            raise TagMismatchException('Error in line %s of file %s\n' % (line_number, self.filepath))
        if leading_spaces_count != last_starttag_col_num and last_starttag_line_num != line_number:
            error_message = '%s --> Indentation for end tag %s on line %s does not match the indentation of the start tag %s on line %s ' % (self.filepath, tag, line_number, last_starttag, last_starttag_line_num)
            self.error_messages.append(error_message)
            self.failed = True
        self.indentation_level -= 1

    def handle_data(self, data: str) -> None:
        if False:
            i = 10
            return i + 15
        'Handle indentation level.\n\n        Args:\n            data: str. Contents of HTML file to be parsed.\n        '
        data_lines = data.split('\n')
        opening_block = tuple(['{% block', '{% macro', '{% if', '% for', '% if'])
        ending_block = tuple(['{% end', '{%- end', '% } %>'])
        for data_line in data_lines:
            data_line = data_line.lstrip()
            if data_line.startswith(opening_block):
                self.indentation_level += 1
            elif data_line.startswith(ending_block):
                self.indentation_level -= 1

    def _check_space_between_attributes_and_values(self, tag: str, attr: str, value: str, rendered_text: str, value_in_quotes: bool, attr_pos_mapping: Dict[str, List[int]]) -> None:
        if False:
            while True:
                i = 10
        'Checks if there are any spaces between attributes and their value.\n\n        Args:\n            tag: str. The tag name of the HTML line.\n            attr: str. The attribute name in the tag.\n            value: str. The value of the attribute.\n            rendered_text: str. The rendered text of the tag.\n            value_in_quotes: bool. Whether the given attribute value\n                is in double quotes.\n            attr_pos_mapping: dict. Mapping between attribute and their\n                starting positions in the tag.\n        '
        (line_number, _) = self.getpos()
        if attr not in attr_pos_mapping:
            attr_positions = []
            for match in re.finditer(re.escape(attr), rendered_text.lower()):
                (start, end) = (match.start(), match.end())
                if rendered_text[start - 1] in [' ', '"'] and rendered_text[end] in [' ', '=']:
                    attr_positions.append(start)
            attr_pos_mapping[attr] = attr_positions
        attr_pos = attr_pos_mapping[attr].pop(0)
        rendered_attr_name = rendered_text[attr_pos:attr_pos + len(attr)]
        attr_val_structure = '{}="{}"' if value_in_quotes else '{}={}'
        expected_attr_assignment = attr_val_structure.format(rendered_attr_name, value)
        if not rendered_text.startswith(expected_attr_assignment, attr_pos):
            self.failed = True
            error_message = '%s --> Attribute %s for tag %s on line %s has unwanted white spaces around it' % (self.filepath, attr, tag, line_number)
            self.error_messages.append(error_message)

class HTMLLintChecksManager(linter_utils.BaseLinter):
    """Manages all the HTML linting functions."""

    def __init__(self, files_to_lint: List[str], file_cache: pre_commit_linter.FileCache) -> None:
        if False:
            while True:
                i = 10
        'Constructs a HTMLLintChecksManager object.\n\n        Args:\n            files_to_lint: list(str). A list of filepaths to lint.\n            file_cache: object(FileCache). Provides thread-safe access to cached\n                file content.\n        '
        self.files_to_lint = files_to_lint
        self.file_cache = file_cache

    @property
    def html_filepaths(self) -> List[str]:
        if False:
            print('Hello World!')
        'Return all html filepaths.'
        return self.files_to_lint

    @property
    def all_filepaths(self) -> List[str]:
        if False:
            return 10
        'Return all filepaths.'
        return self.html_filepaths

    def check_html_tags_and_attributes(self) -> concurrent_task_utils.TaskResult:
        if False:
            print('Hello World!')
        'This function checks the indentation of lines in HTML files.\n\n        Returns:\n            TaskResult. A TaskResult object representing the result of the lint\n            check.\n\n        Raises:\n            TagMismatchException. Proper identation absent in specified\n                html file.\n        '
        html_files_to_lint = self.html_filepaths
        failed = False
        error_messages = []
        name = 'HTML tag and attribute'
        for filepath in html_files_to_lint:
            file_content = self.file_cache.read(filepath)
            file_lines = self.file_cache.readlines(filepath)
            parser = CustomHTMLParser(filepath, file_lines)
            parser.feed(file_content)
            if len(parser.tag_stack) != 0:
                raise TagMismatchException('Error in file %s\n' % filepath)
            if parser.failed:
                error_messages.extend(parser.error_messages)
                failed = True
        return concurrent_task_utils.TaskResult(name, failed, error_messages, error_messages)

    def perform_all_lint_checks(self) -> List[concurrent_task_utils.TaskResult]:
        if False:
            while True:
                i = 10
        'Perform all the lint checks and returns the messages returned by all\n        the checks.\n\n        Returns:\n            list(TaskResult). A list of TaskResult objects representing the\n            results of the lint checks.\n        '
        if not self.all_filepaths:
            return [concurrent_task_utils.TaskResult('HTML lint', False, [], ['There are no HTML files to lint.'])]
        return [self.check_html_tags_and_attributes()]

class ThirdPartyHTMLLintChecksManager(linter_utils.BaseLinter):
    """Manages all the HTML linting functions."""

    def __init__(self, files_to_lint: List[str]) -> None:
        if False:
            while True:
                i = 10
        'Constructs a ThirdPartyHTMLLintChecksManager object.\n\n        Args:\n            files_to_lint: list(str). A list of filepaths to lint.\n        '
        super().__init__()
        self.files_to_lint = files_to_lint

    @property
    def html_filepaths(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Return other filepaths.'
        return self.files_to_lint

    @property
    def all_filepaths(self) -> List[str]:
        if False:
            return 10
        'Return all filepaths.'
        return self.html_filepaths

    @staticmethod
    def _get_trimmed_error_output(html_lint_output: str) -> str:
        if False:
            print('Hello World!')
        'Remove extra bits from htmllint error messages.\n\n        Args:\n            html_lint_output: str. Output returned by the html linter.\n\n        Returns:\n            str. A string with the trimmed error messages.\n        '
        trimmed_error_messages = []
        html_output_lines = html_lint_output.split('\n')
        empty_string_present = html_output_lines[-1] == ''
        htmllint_present = html_output_lines[-2].startswith('[htmllint]')
        if empty_string_present and htmllint_present:
            html_output_lines = html_output_lines[:-2]
        for line in html_output_lines:
            trimmed_error_messages.append(line)
        return '\n'.join(trimmed_error_messages) + '\n'

    def lint_html_files(self) -> concurrent_task_utils.TaskResult:
        if False:
            return 10
        'This function is used to check HTML files for linting errors.\n\n        Returns:\n            TaskResult. A TaskResult object representing the result of the lint\n            check.\n        '
        node_path = os.path.join(common.NODE_PATH, 'bin', 'node')
        htmllint_path = os.path.join('node_modules', 'htmllint-cli', 'bin', 'cli.js')
        failed = False
        name = 'HTMLLint'
        error_messages = []
        full_error_messages = []
        htmllint_cmd_args = [node_path, htmllint_path, '--rc=.htmllintrc']
        proc_args = htmllint_cmd_args + self.html_filepaths
        proc = subprocess.Popen(proc_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (encoded_linter_stdout, _) = proc.communicate()
        linter_stdout = encoded_linter_stdout.decode('utf-8')
        error_count = [int(s) for s in linter_stdout.split() if s.isdigit()][-2]
        if error_count:
            failed = True
            full_error_messages.append(linter_stdout)
            error_messages.append(self._get_trimmed_error_output(linter_stdout))
        return concurrent_task_utils.TaskResult(name, failed, error_messages, full_error_messages)

    def perform_all_lint_checks(self) -> List[concurrent_task_utils.TaskResult]:
        if False:
            i = 10
            return i + 15
        'Perform all the lint checks and returns the messages returned by all\n        the checks.\n\n        Returns:\n            list(TaskResult). A list of TaskResult objects representing the\n            results of the lint checks.\n        '
        if not self.all_filepaths:
            return [concurrent_task_utils.TaskResult('HTML lint', False, [], ['There are no HTML files to lint.'])]
        return [self.lint_html_files()]

def get_linters(files_to_lint: List[str], file_cache: pre_commit_linter.FileCache) -> Tuple[HTMLLintChecksManager, ThirdPartyHTMLLintChecksManager]:
    if False:
        return 10
    'Creates HTMLLintChecksManager and ThirdPartyHTMLLintChecksManager\n        objects and returns them.\n\n    Args:\n        files_to_lint: list(str). A list of filepaths to lint.\n        file_cache: object(FileCache). Provides thread-safe access to cached\n            file content.\n\n    Returns:\n        tuple(HTMLLintChecksManager, ThirdPartyHTMLLintChecksManager). A 2-tuple\n        of custom and third_party linter objects.\n    '
    custom_linter = HTMLLintChecksManager(files_to_lint, file_cache)
    third_party_linter = ThirdPartyHTMLLintChecksManager(files_to_lint)
    return (custom_linter, third_party_linter)