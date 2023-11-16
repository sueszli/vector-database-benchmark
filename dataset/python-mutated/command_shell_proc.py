"""
This module provides pre-/post-processors for the mod:`behave4cmd0.command_shell`.
"""
from __future__ import absolute_import, print_function
import re
import sys
from six import string_types

def posixpath_normpath(filename):
    if False:
        print('Hello World!')
    if not filename:
        return filename
    return filename.replace('\\', '/').replace('//', '/')

class LineProcessor(object):
    """Function-like object that may perform text-line transformations."""

    def __init__(self, marker=None):
        if False:
            print('Hello World!')
        self.marker = marker

    def reset(self):
        if False:
            return 10
        pass

    def __call__(self, text):
        if False:
            print('Hello World!')
        return text

class TracebackLineNormalizer(LineProcessor):
    """Line processor that tries to normalize path lines in a traceback dump."""
    marker = 'Traceback (most recent call last):'
    file_pattern = re.compile('\\s\\s+File "(?P<path>.*)", line .*')

    def __init__(self):
        if False:
            print('Hello World!')
        super(TracebackLineNormalizer, self).__init__(self.marker)
        self.traceback_section = False

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.traceback_section = False

    def __call__(self, line):
        if False:
            i = 10
            return i + 15
        'Process a line and optionally transform it.\n\n        :param line: line to process (as text)\n        :return: Same line or transformed/normalized line (as text).\n        '
        marker = self.marker
        stripped_line = line.strip()
        if marker == stripped_line:
            assert not self.traceback_section
            self.traceback_section = True
        elif self.traceback_section:
            matched = self.file_pattern.match(line)
            if matched:
                filename = matched.groups()[0]
                new_filename = posixpath_normpath(filename)
                if new_filename != filename:
                    line = line.replace(filename, new_filename)
            elif not stripped_line or line[0].isalpha():
                self.traceback_section = False
        return line

class ExceptionWithPathNormalizer(LineProcessor):
    """Normalize filename path in Exception line (for Windows)."""
    problematic_path_patterns = ['ConfigError: No steps directory in "(?P<path>.*)"', 'ParserError: Failed to parse "(?P<path>.*)"', "Error: [Errno 2] No such file or directory: '(?P<path>.*)'"]

    def __init__(self, pattern, marker_text=None):
        if False:
            return 10
        super(ExceptionWithPathNormalizer, self).__init__(marker_text)
        self.pattern = re.compile(pattern, re.UNICODE)
        self.marker = marker_text

    def __call__(self, line):
        if False:
            while True:
                i = 10
        matched = self.pattern.search(line)
        if matched:
            filename = matched.groupdict()['path']
            new_filename = posixpath_normpath(filename)
            if new_filename != filename:
                line = line.replace(filename, new_filename)
        return line

class CommandPostProcessor(object):
    """Syntactic sugar to mark a command post-processor."""

class CommandOutputProcessor(CommandPostProcessor):
    """Abstract base class functionality for a CommandPostProcessor that
    post-processes the output of a command.
    """
    enabled = True
    output_parts = ('stderr', 'stdout')

    def __init__(self, enabled=None, output_parts=None):
        if False:
            print('Hello World!')
        if enabled is None:
            enabled = self.__class__.enabled
        if output_parts is None:
            output_parts = self.__class__.output_parts
        self.enabled = enabled
        self.output_parts = output_parts

    def matches_output(self, text):
        if False:
            for i in range(10):
                print('nop')
        'Abstract method that should be overwritten.'
        return False

    def process_output(self, text):
        if False:
            for i in range(10):
                print('nop')
        'Abstract method that should be overwritten.'
        changed = False
        return (changed, text)

    def __call__(self, command_result):
        if False:
            print('Hello World!')
        'Core functionality of command output processor.\n\n        :param command_result:  As value object w/ command execution details.\n        :return: Command result\n        '
        if not self.enabled:
            return command_result
        changes = 0
        for output_name in self.output_parts:
            output = getattr(command_result, output_name)
            if output and self.matches_output(output):
                (changed, new_output) = self.process_output(output)
                if changed:
                    changes += 1
                    setattr(command_result, output_name, new_output)
        if changes:
            command_result._output = None
        return command_result

class LineCommandOutputProcessor(CommandOutputProcessor):
    """Provides functionality to process text in line-oriented way by using
    a number of line processors. The line processors perform the actual work
    for transforming/normalizing the text.
    """
    enabled = True
    line_processors = [TracebackLineNormalizer()]

    def __init__(self, line_processors=None):
        if False:
            for i in range(10):
                print('nop')
        if line_processors is None:
            line_processors = self.__class__.line_processors
        super(LineCommandOutputProcessor, self).__init__(self.enabled)
        self.line_processors = line_processors
        self.markers = [p.marker for p in self.line_processors if p.marker]

    def matches_output(self, text):
        if False:
            while True:
                i = 10
        'Indicates it text contains sections of interest.\n        :param text:    Text to inspect (as string).\n        :return: True, if text contains Traceback sections. False, otherwise.\n        '
        if self.markers:
            for marker in self.markers:
                if marker in text:
                    return True
        return False

    def process_output(self, text):
        if False:
            print('Hello World!')
        'Normalizes multi-line text by applying the line processors.\n\n        :param text:    Text to process (as string).\n        :return: Tuple (changed : bool, new_text : string)\n        '
        new_lines = []
        changed = False
        for line_processor in self.line_processors:
            line_processor.reset()
        for line in text.splitlines():
            original_line = line
            for line_processor in self.line_processors:
                line = line_processor(line)
            if line != original_line:
                changed = True
            new_lines.append(line)
        if changed:
            text = '\n'.join(new_lines) + '\n'
        return (changed, text)

class TextProcessor(CommandOutputProcessor):
    """Provides an adapter that uses an :class:`CommandOutputProcessor`
    as text processor (normalizer).
    """

    def __init__(self, command_output_processor):
        if False:
            while True:
                i = 10
        self.command_output_processor = command_output_processor
        self.enabled = self.command_output_processor.enabled
        self.output_parts = self.command_output_processor.output_parts

    def process_output(self, text):
        if False:
            while True:
                i = 10
        return self.command_output_processor.process_output(text)

    def __call__(self, command_result):
        if False:
            i = 10
            return i + 15
        if isinstance(command_result, string_types):
            text = command_result
            return self.command_output_processor.process_output(text)[1]
        else:
            return self.command_output_processor(command_result)

class BehaveWinCommandOutputProcessor(LineCommandOutputProcessor):
    """Command output post-processor for :mod:`behave` on Windows platform.
    Mostly, normalizes windows paths in output and exceptions to conform to
    POSIX path conventions.
    """
    enabled = sys.platform.startswith('win') or True
    line_processors = [TracebackLineNormalizer(), ExceptionWithPathNormalizer("ConfigError: No steps directory in '(?P<path>.*)'", 'ConfigError: No steps directory in'), ExceptionWithPathNormalizer('ParserError: Failed to parse "(?P<path>.*)"', 'ParserError: Failed to parse'), ExceptionWithPathNormalizer("No such file or directory: '(?P<path>.*)'", '[Errno 2] No such file or directory:'), ExceptionWithPathNormalizer('^\\s*File "(?P<path>.*)", line \\d+, in ', 'File "')]