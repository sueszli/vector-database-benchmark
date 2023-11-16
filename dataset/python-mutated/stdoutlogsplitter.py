import re
from datetime import datetime
from .loggerhelper import Message, write_to_console

class StdoutLogSplitter:
    """Splits messages logged through stdout (or stderr) into Message objects"""
    _split_from_levels = re.compile('^(?:\\*(TRACE|DEBUG|INFO|CONSOLE|HTML|WARN|ERROR)(:\\d+(?:\\.\\d+)?)?\\*)', re.MULTILINE)

    def __init__(self, output):
        if False:
            i = 10
            return i + 15
        self._messages = list(self._get_messages(output.strip()))

    def _get_messages(self, output):
        if False:
            i = 10
            return i + 15
        for (level, timestamp, msg) in self._split_output(output):
            if level == 'CONSOLE':
                write_to_console(msg.lstrip())
                level = 'INFO'
            if timestamp:
                timestamp = datetime.fromtimestamp(float(timestamp[1:]) / 1000)
            yield Message(msg.strip(), level, timestamp=timestamp)

    def _split_output(self, output):
        if False:
            for i in range(10):
                print('nop')
        tokens = self._split_from_levels.split(output)
        tokens = self._add_initial_level_and_time_if_needed(tokens)
        for i in range(0, len(tokens), 3):
            yield tokens[i:i + 3]

    def _add_initial_level_and_time_if_needed(self, tokens):
        if False:
            i = 10
            return i + 15
        if self._output_started_with_level(tokens):
            return tokens[1:]
        return ['INFO', None] + tokens

    def _output_started_with_level(self, tokens):
        if False:
            i = 10
            return i + 15
        return tokens[0] == ''

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self._messages)

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._messages)

    def __getitem__(self, item):
        if False:
            i = 10
            return i + 15
        return self._messages[item]