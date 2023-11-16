from typing import Optional
from typing import Union
import click
UNITS = {'': 1, 'B': 1, 'KIB': 2 ** 10, 'MIB': 2 ** 20, 'GIB': 2 ** 30, 'TIB': 2 ** 40, 'KB': 10 ** 3, 'MB': 10 ** 6, 'GB': 10 ** 9, 'TB': 10 ** 12}

def parse_size(input: str) -> int:
    if False:
        for i in range(10):
            print('nop')
    import re
    s = input.upper()
    s = re.sub('([BKMGT][A-Z]*)', ' \\1', s)
    tokens = [sub.strip() for sub in s.split()]
    n = len(tokens)
    if n == 1:
        number = tokens[0]
        unit = ''
    elif n == 2:
        (number, unit) = tokens
    else:
        raise ValueError(f"Invalid representation for a number of bytes: '{input}'")
    if unit in UNITS:
        return int(float(number) * UNITS[unit])
    else:
        raise ValueError(f"Invalid representation for a number of bytes: '{input}'")

class ByteSizeType(click.ParamType):
    name = 'BYTES'

    def convert(self, value: Union[None, str, int], _param: Optional[click.Parameter], ctx: Optional[click.Context]) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        try:
            return parse_size(value) if isinstance(value, str) else value if isinstance(value, int) else None
        except ValueError as ex:
            raise click.exceptions.UsageError(*ex.args)