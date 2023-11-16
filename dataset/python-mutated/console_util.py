from __future__ import print_function
import numpy as np

def fmt_row(width, row, header=False):
    if False:
        while True:
            i = 10
    "\n    fits a list of items to at least a certain length\n\n    :param width: (int) the minimum width of the string\n    :param row: ([Any]) a list of object you wish to get the string representation\n    :param header: (bool) whether or not to return the string as a header\n    :return: (str) the string representation of all the elements in 'row', of length >= 'width'\n    "
    out = ' | '.join((fmt_item(x, width) for x in row))
    if header:
        out = out + '\n' + '-' * len(out)
    return out

def fmt_item(item, min_width):
    if False:
        print('Hello World!')
    "\n    fits items to a given string length\n\n    :param item: (Any) the item you wish to get the string representation\n    :param min_width: (int) the minimum width of the string\n    :return: (str) the string representation of 'x' of length >= 'l'\n    "
    if isinstance(item, np.ndarray):
        assert item.ndim == 0
        item = item.item()
    if isinstance(item, (float, np.float32, np.float64)):
        value = abs(item)
        if (value < 0.0001 or value > 10000.0) and value > 0:
            rep = '%7.2e' % item
        else:
            rep = '%7.5f' % item
    else:
        rep = str(item)
    return ' ' * (min_width - len(rep)) + rep
COLOR_TO_NUM = dict(gray=30, red=31, green=32, yellow=33, blue=34, magenta=35, cyan=36, white=37, crimson=38)

def colorize(string, color, bold=False, highlight=False):
    if False:
        return 10
    '\n    Colorize, bold and/or highlight a string for terminal print\n\n    :param string: (str) input string\n    :param color: (str) the color, the lookup table is the dict at console_util.color2num\n    :param bold: (bool) if the string should be bold or not\n    :param highlight: (bool) if the string should be highlighted or not\n    :return: (str) the stylized output string\n    '
    attr = []
    num = COLOR_TO_NUM[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)