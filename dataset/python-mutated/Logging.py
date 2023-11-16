from datetime import datetime
from collections import Counter
import json
import io
import logging
import logging.config

class CounterHandler(logging.Handler):
    """
    A logging handler which counts the number of calls
    for each logging level.
    """
    _call_counter = Counter()

    @classmethod
    def reset(cls):
        if False:
            return 10
        '\n        Reset the counter to 0 for all levels\n        '
        cls._call_counter.clear()

    @classmethod
    def emit(cls, record):
        if False:
            for i in range(10):
                print('nop')
        cls._call_counter[record.levelname] += 1

    @classmethod
    def get_num_calls_for_level(cls, level):
        if False:
            while True:
                i = 10
        '\n        Returns the number of calls registered for a given log level.\n        '
        return cls._call_counter[level]

def configure_logging(color=True):
    if False:
        print('Hello World!')
    '\n    Configures the logging with hard coded dictionary.\n    '
    import sys
    CounterHandler.reset()
    logging.config.dictConfig({'version': 1, 'handlers': {'colored': {'class': 'logging.StreamHandler', 'formatter': 'colored' if color else 'plain', 'stream': sys.stderr}, 'counter': {'class': 'coalib.output.Logging.CounterHandler'}}, 'root': {'level': 'DEBUG', 'handlers': ['colored', 'counter']}, 'formatters': {'colored': {'()': 'colorlog.ColoredFormatter', 'format': '%(log_color)s[%(levelname)s]%(reset)s[%(asctime)s] %(message)s', 'datefmt': '%X', 'log_colors': {'ERROR': 'red', 'WARNING': 'yellow', 'INFO': 'blue', 'DEBUG': 'green'}}, 'plain': {'format': '[%(levelname)s][%(asctime)s] %(message)s', 'datefmt': '%X'}}})

def configure_json_logging():
    if False:
        i = 10
        return i + 15
    '\n    Configures logging for JSON.\n    :return: Returns a ``StringIO`` that captures the logs as JSON.\n    '
    stream = io.StringIO()
    CounterHandler.reset()
    logging.config.dictConfig({'version': 1, 'handlers': {'json': {'class': 'logging.StreamHandler', 'formatter': 'json', 'stream': stream}, 'counter': {'class': 'coalib.output.Logging.CounterHandler'}}, 'root': {'level': 'DEBUG', 'handlers': ['json', 'counter']}, 'formatters': {'json': {'()': 'coalib.output.Logging.JSONFormatter'}}})
    return stream

class JSONFormatter(logging.Formatter):
    """
    JSON formatter for python logging.
    """

    @staticmethod
    def format(record):
        if False:
            while True:
                i = 10
        message = {'timestamp': datetime.utcfromtimestamp(record.created).isoformat(), 'message': record.getMessage(), 'level': record.levelname}
        return json.dumps(message)