from pprint import pprint
import click
from mitmproxy.io import tnetstring

def read_tnetstring(input):
    if False:
        print('Hello World!')
    if not input.read(1):
        return None
    else:
        input.seek(-1, 1)
    return tnetstring.load(input)

@click.command()
@click.argument('input', type=click.File('rb'))
def inspect(input):
    if False:
        i = 10
        return i + 15
    '\n    pretty-print a dumpfile\n    '
    while True:
        data = read_tnetstring(input)
        if not data:
            break
        pprint(data)
if __name__ == '__main__':
    inspect()