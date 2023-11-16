""" Work with ReST documentations.

This e.g. creates PDF documentations during release and tables from data
for the web site, e.g. downloads.

"""
import functools

def makeTable(grid):
    if False:
        while True:
            i = 10
    'Create a REST table.'

    def makeSeparator(num_cols, col_width, header_flag):
        if False:
            i = 10
            return i + 15
        if header_flag == 1:
            return num_cols * ('+' + col_width * '=') + '+\n'
        else:
            return num_cols * ('+' + col_width * '-') + '+\n'

    def normalizeCell(string, length):
        if False:
            i = 10
            return i + 15
        return string + (length - len(string)) * ' '
    cell_width = 2 + max(functools.reduce(lambda x, y: x + y, [[len(item) for item in row] for row in grid], []))
    num_cols = len(grid[0])
    rst = makeSeparator(num_cols, cell_width, 0)
    header_flag = 1
    for row in grid:
        rst = rst + '| ' + '| '.join([normalizeCell(x, cell_width - 1) for x in row]) + '|\n'
        rst = rst + makeSeparator(num_cols, cell_width, header_flag)
        header_flag = 0
    return rst