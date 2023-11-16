import json
import logging
import logging.config
import logging.handlers
import os
import queue
JSON_LOGGING = os.environ.get('JSON_LOGGING', 'false').lower() == 'true'
CHAT = 29
logging.addLevelName(CHAT, 'CHAT')
RESET_SEQ: str = '\x1b[0m'
COLOR_SEQ: str = '\x1b[1;%dm'
BOLD_SEQ: str = '\x1b[1m'
UNDERLINE_SEQ: str = '\x1b[04m'
ORANGE: str = '\x1b[33m'
YELLOW: str = '\x1b[93m'
WHITE: str = '\x1b[37m'
BLUE: str = '\x1b[34m'
LIGHT_BLUE: str = '\x1b[94m'
RED: str = '\x1b[91m'
GREY: str = '\x1b[90m'
GREEN: str = '\x1b[92m'
EMOJIS: dict[str, str] = {'DEBUG': '🐛', 'INFO': '📝', 'CHAT': '💬', 'WARNING': '⚠️', 'ERROR': '❌', 'CRITICAL': '💥'}
KEYWORD_COLORS: dict[str, str] = {'DEBUG': WHITE, 'INFO': LIGHT_BLUE, 'CHAT': GREEN, 'WARNING': YELLOW, 'ERROR': ORANGE, 'CRITICAL': RED}

class JsonFormatter(logging.Formatter):

    def format(self, record):
        if False:
            i = 10
            return i + 15
        return json.dumps(record.__dict__)

def formatter_message(message: str, use_color: bool=True) -> str:
    if False:
        print('Hello World!')
    '\n    Syntax highlight certain keywords\n    '
    if use_color:
        message = message.replace('$RESET', RESET_SEQ).replace('$BOLD', BOLD_SEQ)
    else:
        message = message.replace('$RESET', '').replace('$BOLD', '')
    return message

def format_word(message: str, word: str, color_seq: str, bold: bool=False, underline: bool=False) -> str:
    if False:
        print('Hello World!')
    '\n    Surround the fiven word with a sequence\n    '
    replacer = color_seq + word + RESET_SEQ
    if underline:
        replacer = UNDERLINE_SEQ + replacer
    if bold:
        replacer = BOLD_SEQ + replacer
    return message.replace(word, replacer)

class ConsoleFormatter(logging.Formatter):
    """
    This Formatted simply colors in the levelname i.e 'INFO', 'DEBUG'
    """

    def __init__(self, fmt: str, datefmt: str=None, style: str='%', use_color: bool=True):
        if False:
            while True:
                i = 10
        super().__init__(fmt, datefmt, style)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if False:
            print('Hello World!')
        '\n        Format and highlight certain keywords\n        '
        rec = record
        levelname = rec.levelname
        if self.use_color and levelname in KEYWORD_COLORS:
            levelname_color = KEYWORD_COLORS[levelname] + levelname + RESET_SEQ
            rec.levelname = levelname_color
        rec.name = f'{GREY}{rec.name:<15}{RESET_SEQ}'
        rec.msg = KEYWORD_COLORS[levelname] + EMOJIS[levelname] + '  ' + rec.msg + RESET_SEQ
        return logging.Formatter.format(self, rec)

class ForgeLogger(logging.Logger):
    """
    This adds extra logging functions such as logger.trade and also
    sets the logger to use the custom formatter
    """
    CONSOLE_FORMAT: str = '[%(asctime)s] [$BOLD%(name)-15s$RESET] [%(levelname)-8s]\t%(message)s'
    FORMAT: str = '%(asctime)s %(name)-15s %(levelname)-8s %(message)s'
    COLOR_FORMAT: str = formatter_message(CONSOLE_FORMAT, True)
    JSON_FORMAT: str = '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'

    def __init__(self, name: str, logLevel: str='DEBUG'):
        if False:
            for i in range(10):
                print('nop')
        logging.Logger.__init__(self, name, logLevel)
        queue_handler = logging.handlers.QueueHandler(queue.Queue(-1))
        json_formatter = logging.Formatter(self.JSON_FORMAT)
        queue_handler.setFormatter(json_formatter)
        self.addHandler(queue_handler)
        if JSON_LOGGING:
            console_formatter = JsonFormatter()
        else:
            console_formatter = ConsoleFormatter(self.COLOR_FORMAT)
        console = logging.StreamHandler()
        console.setFormatter(console_formatter)
        self.addHandler(console)

    def chat(self, role: str, openai_repsonse: dict, messages=None, *args, **kws):
        if False:
            print('Hello World!')
        '\n        Parse the content, log the message and extract the usage into prometheus metrics\n        '
        role_emojis = {'system': '🖥️', 'user': '👤', 'assistant': '🤖', 'function': '⚙️'}
        if self.isEnabledFor(CHAT):
            if messages:
                for message in messages:
                    self._log(CHAT, f"{role_emojis.get(message['role'], '🔵')}: {message['content']}")
            else:
                response = json.loads(openai_repsonse)
                self._log(CHAT, f"{role_emojis.get(role, '🔵')}: {response['choices'][0]['message']['content']}")

class QueueLogger(logging.Logger):
    """
    Custom logger class with queue
    """

    def __init__(self, name: str, level: int=logging.NOTSET):
        if False:
            while True:
                i = 10
        super().__init__(name, level)
        queue_handler = logging.handlers.QueueHandler(queue.Queue(-1))
        self.addHandler(queue_handler)
logging_config: dict = dict(version=1, formatters={'console': {'()': ConsoleFormatter, 'format': ForgeLogger.COLOR_FORMAT}}, handlers={'h': {'class': 'logging.StreamHandler', 'formatter': 'console', 'level': logging.INFO}}, root={'handlers': ['h'], 'level': logging.INFO}, loggers={'autogpt': {'handlers': ['h'], 'level': logging.INFO, 'propagate': False}})

def setup_logger():
    if False:
        while True:
            i = 10
    '\n    Setup the logger with the specified format\n    '
    logging.config.dictConfig(logging_config)