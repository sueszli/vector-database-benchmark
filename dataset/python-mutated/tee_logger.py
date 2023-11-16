"""
A logger that maintains logs of both stdout and stderr when models are run.
"""
from typing import TextIO
import os

class TeeLogger:
    """
    This class is an attempt to maintain logs of both stdout and stderr for when models are run.
    To use this class, at the beginning of your script insert these lines::

        sys.stdout = TeeLogger("stdout.log", sys.stdout)
        sys.stderr = TeeLogger("stdout.log", sys.stderr)
    """

    def __init__(self, filename: str, terminal: TextIO) -> None:
        if False:
            print('Hello World!')
        self.terminal = terminal
        parent_directory = os.path.dirname(filename)
        os.makedirs(parent_directory, exist_ok=True)
        self.log = open(filename, 'a')

    def write(self, message):
        if False:
            return 10
        self.terminal.write(message)
        if '\x08' in message:
            message = message.replace('\x08', '')
            if not message or message[-1] != '\n':
                message += '\n'
        self.log.write(message)

    def flush(self):
        if False:
            return 10
        self.terminal.flush()
        self.log.flush()