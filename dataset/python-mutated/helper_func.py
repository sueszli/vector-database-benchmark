def make_table(header, content, column_width=None):
    if False:
        i = 10
        return i + 15
    '\n    Input:\n    header -> List[str]: table header\n    content -> List[List[str]]: table content\n    column_width -> int: table column width; set to None for dynamically calculated widths\n    \n    Output:\n    table_str -> str: well-formatted string for the table\n    '
    table_str = ''
    (len_column, len_row) = (len(header), len(content) + 1)
    if column_width is None:
        lens = [[len(str(h)) for h in header]]
        lens += [[len(str(x)) for x in row] for row in content]
        column_widths = [max(c) + 3 for c in zip(*lens)]
    else:
        column_widths = [column_width] * len_column
    table_str += '=' * (sum(column_widths) + 1) + '\n'
    table_str += '|'
    for (i, item) in enumerate(header):
        table_str += ' ' + str(item).ljust(column_widths[i] - 2) + '|'
    table_str += '\n'
    table_str += '-' * (sum(column_widths) + 1) + '\n'
    for line in content:
        table_str += '|'
        for (i, item) in enumerate(line):
            table_str += ' ' + str(item).ljust(column_widths[i] - 2) + '|'
        table_str += '\n'
    table_str += '=' * (sum(column_widths) + 1) + '\n'
    return table_str