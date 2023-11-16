from datetime import datetime
from coalib.output.printers.LOG_LEVEL import LOG_LEVEL

class LogMessage:

    def __init__(self, log_level, *messages, delimiter=' ', timestamp=None):
        if False:
            print('Hello World!')
        if log_level not in LOG_LEVEL.reverse:
            raise ValueError('log_level has to be a valid LOG_LEVEL.')
        str_messages = [str(message) for message in messages]
        self.message = str(delimiter).join(str_messages).rstrip()
        if self.message == '':
            raise ValueError('Empty log messages are not allowed.')
        self.log_level = log_level
        self.timestamp = datetime.today() if timestamp is None else timestamp

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        log_level = LOG_LEVEL.reverse.get(self.log_level, 'ERROR')
        return f'[{log_level}] {self.message}'

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, LogMessage) and other.log_level == self.log_level and (other.message == self.message)

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self.__eq__(other)

    def to_string_dict(self):
        if False:
            i = 10
            return i + 15
        '\n        Makes a dictionary which has all keys and values as strings and\n        contains all the data that the LogMessage has.\n\n        :return: Dictionary with keys and values as string.\n        '
        retval = {}
        retval['message'] = str(self.message)
        retval['timestamp'] = '' if self.timestamp is None else self.timestamp.isoformat()
        retval['log_level'] = str(LOG_LEVEL.reverse.get(self.log_level, ''))
        return retval