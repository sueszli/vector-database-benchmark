import re
from typing import List
TRACEBACK_PATTERN = 'Traceback (most recent call last)'

class LogAggregator:

    def __init__(self, log: str):
        if False:
            print('Hello World!')
        self.log = log

    def compute_crash_pattern(self) -> str:
        if False:
            while True:
                i = 10
        stack_trace = LogAggregator._compute_stack_trace(self.log.splitlines())
        return LogAggregator._compute_signature(stack_trace)[:4000]

    @staticmethod
    def _compute_signature(stack_trace: List[str]) -> str:
        if False:
            while True:
                i = 10
        '\n        Compute signature pattern from stack trace, by remove factors such as date,\n        time, temp directory, line numbers, etc. This help to aggregate similar logs\n        into same bug patterns\n        '
        massaged_trace = []
        for line in stack_trace:
            line = re.sub('[a-z0-9]{10,}', '', line.strip())
            line = re.sub('\\d', '', line)
            if line == 'Traceback (most recent call last):':
                continue
            file_line = re.search('File "(.*)", (.*)', line)
            if file_line:
                line = f"{file_line.group(1).split('/')[-1]}{file_line.group(2)}"
            massaged_trace.append(line)
        return ''.join(massaged_trace)

    @staticmethod
    def _compute_stack_trace(logs: List[str]) -> List[str]:
        if False:
            while True:
                i = 10
        '\n        Extract stack trace pattern from the logs. Stack trace pattern often matches\n        the following:\n        ERROR ...\n        Traceback (most recent call last):\n            File "...", line ..., in ...\n            ...\n        Exception: exception error\n        '
        error_stacktrace = []
        stacktrace = []
        i = 0
        while i < len(logs):
            stack = []
            trace = error_stacktrace
            if 'ERROR' in logs[i]:
                stack.append(logs[i])
                next = i + 1
                if i + 1 < len(logs) and TRACEBACK_PATTERN in logs[i + 1]:
                    stack.append(logs[i + 1])
                    next = i + 2
            elif TRACEBACK_PATTERN in logs[i]:
                stack.append(logs[i])
                trace = stacktrace
                next = i + 1
            else:
                i = i + 1
                continue
            while next < len(logs):
                if logs[next].startswith((' ', '\t')):
                    stack.append(logs[next])
                    next = next + 1
                else:
                    break
            if next < len(logs):
                stack.append(logs[next])
            if stack:
                trace.append(stack)
            i = next + 1
        if error_stacktrace:
            return error_stacktrace[-1]
        if stacktrace:
            return stacktrace[-1]
        return []