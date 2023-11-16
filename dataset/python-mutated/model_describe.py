"""
Provides textual descriptions for :mod:`behave.model` elements.
"""
from __future__ import absolute_import
from six.moves import range
from six.moves import zip
from behave.textutil import indent

def escape_cell(cell):
    if False:
        i = 10
        return i + 15
    '\n    Escape table cell contents.\n    :param cell:  Table cell (as unicode string).\n    :return: Escaped cell (as unicode string).\n    '
    cell = cell.replace(u'\\', u'\\\\')
    cell = cell.replace(u'\n', u'\\n')
    cell = cell.replace(u'|', u'\\|')
    return cell

def escape_triple_quotes(text):
    if False:
        return 10
    '\n    Escape triple-quotes, used for multi-line text/doc-strings.\n    '
    return text.replace(u'"""', u'\\"\\"\\"')

class ModelDescriptor(object):

    @staticmethod
    def describe_table(table, indentation=None):
        if False:
            print('Hello World!')
        '\n        Provide a textual description of the table (as used w/ Gherkin).\n\n        :param table:  Table to use (as :class:`behave.model.Table`)\n        :param indentation:  Line prefix to use (as string, if any).\n        :return: Textual table description (as unicode string).\n        '
        cell_lengths = []
        all_rows = [table.headings] + table.rows
        for row in all_rows:
            lengths = [len(escape_cell(c)) for c in row]
            cell_lengths.append(lengths)
        max_lengths = []
        for col in range(0, len(cell_lengths[0])):
            max_lengths.append(max([c[col] for c in cell_lengths]))
        lines = []
        for (r, row) in enumerate(all_rows):
            line = u'|'
            for (c, (cell, max_length)) in enumerate(zip(row, max_lengths)):
                pad_size = max_length - cell_lengths[r][c]
                line += u' %s%s |' % (escape_cell(cell), ' ' * pad_size)
            line += u'\n'
            lines.append(line)
        if indentation:
            return indent(lines, indentation)
        return u''.join(lines)

    @staticmethod
    def describe_docstring(doc_string, indentation=None):
        if False:
            while True:
                i = 10
        '\n        Provide a textual description of the multi-line text/triple-quoted\n        doc-string (as used w/ Gherkin).\n\n        :param doc_string:  Multi-line text to use.\n        :param indentation:  Line prefix to use (as string, if any).\n        :return: Textual table description (as unicode string).\n        '
        text = escape_triple_quotes(doc_string)
        text = u'"""\n' + text + '\n"""\n'
        if indentation:
            text = indent(text, indentation)
        return text

class ModelPrinter(ModelDescriptor):

    def __init__(self, stream):
        if False:
            return 10
        super(ModelPrinter, self).__init__()
        self.stream = stream

    def print_table(self, table, indentation=None):
        if False:
            print('Hello World!')
        self.stream.write(self.describe_table(table, indentation))
        self.stream.flush()

    def print_docstring(self, text, indentation=None):
        if False:
            i = 10
            return i + 15
        self.stream.write(self.describe_docstring(text, indentation))
        self.stream.flush()