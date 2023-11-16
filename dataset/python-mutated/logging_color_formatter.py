import logging
import random

def mkcolor(color, bold=False):
    if False:
        print('Hello World!')
    if bold:
        color = f'1;{color}'
    return f'\x1b[{color}m'

class ColoredFormatter(logging.Formatter):
    formats = {logging.DEBUG: ('ℹ️', 34), logging.INFO: ('ℹ️', 37), logging.WARNING: ('⚠️', 33), logging.ERROR: ('⚠️', 31), logging.CRITICAL: ('⚠️', 31)}

    def format(self, record):
        if False:
            i = 10
            return i + 15
        (symbol, level_color) = self.formats.get(record.levelno, ('', 0))
        prefix = f'{symbol}  {mkcolor(level_color, True)}{record.levelname}{mkcolor(0)}'
        if record.name != 'root':
            random.seed(record.name)
            name_color = random.randint(32, 37)
            prefix += f'{mkcolor(name_color, True)} {record.name}{mkcolor(0)}:'
        suffix = f'{mkcolor(2)}{record.module}.{record.funcName}:{record.lineno}{mkcolor(0)}'
        formatter = logging.Formatter(f'%(asctime)s {prefix} %(message)s {suffix}')
        return formatter.format(record)