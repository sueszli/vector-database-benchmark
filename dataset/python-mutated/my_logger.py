from typing import IO, Callable, List
from trashcli.put.core.logs import Level
from trashcli.put.core.logs import LogEntry

class LogData:

    def __init__(self, program_name, verbose):
        if False:
            while True:
                i = 10
        self.program_name = program_name
        self.verbose = verbose

class MyLogger:

    def __init__(self, stderr):
        if False:
            return 10
        self.stderr = stderr

    def debug(self, message, log_data):
        if False:
            print('Hello World!')
        if log_data.verbose > 1:
            self.stderr.write('%s: %s\n' % (log_data.program_name, message))

    def debug_func_result(self, messages_func, log_data):
        if False:
            i = 10
            return i + 15
        if log_data.verbose > 1:
            for line in messages_func():
                self.stderr.write('%s: %s\n' % (log_data.program_name, line))

    def info(self, message, log_data):
        if False:
            return 10
        if log_data.verbose > 0:
            self.stderr.write('%s: %s\n' % (log_data.program_name, message))

    def warning2(self, message, program_name):
        if False:
            while True:
                i = 10
        self.stderr.write('%s: %s\n' % (program_name, message))

    def log_multiple(self, entries, log_data):
        if False:
            print('Hello World!')
        for entry in entries:
            if entry.level == Level.INFO:
                self.info(entry.message, log_data)
            elif entry.level == Level.DEBUG:
                self.debug(entry.message, log_data)
            else:
                raise ValueError('unknown level: %s' % entry.level)