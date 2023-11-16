def print_table(records, fields, formatter):
    if False:
        i = 10
        return i + 15
    formatter.headings(fields)
    for r in records:
        rowdata = [getattr(r, fieldname) for fieldname in fields]
        formatter.row(rowdata)

class TableFormatter:

    def headings(self, headers):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def row(self, rowdata):
        if False:
            print('Hello World!')
        raise NotImplementedError()

class TextTableFormatter(TableFormatter):

    def headings(self, headers):
        if False:
            return 10
        print(' '.join(('%10s' % h for h in headers)))
        print(('-' * 10 + ' ') * len(headers))

    def row(self, rowdata):
        if False:
            i = 10
            return i + 15
        print(' '.join(('%10s' % d for d in rowdata)))

class CSVTableFormatter(TableFormatter):

    def headings(self, headers):
        if False:
            print('Hello World!')
        print(','.join(headers))

    def row(self, rowdata):
        if False:
            return 10
        print(','.join((str(d) for d in rowdata)))

class HTMLTableFormatter(TableFormatter):

    def headings(self, headers):
        if False:
            print('Hello World!')
        print('<tr>', end=' ')
        for h in headers:
            print('<th>%s</th>' % h, end=' ')
        print('</tr>')

    def row(self, rowdata):
        if False:
            for i in range(10):
                print('nop')
        print('<tr>', end=' ')
        for d in rowdata:
            print('<td>%s</td>' % d, end=' ')
        print('</tr>')

def create_formatter(name):
    if False:
        for i in range(10):
            print('nop')
    if name == 'text':
        formatter_cls = TextTableFormatter
    elif name == 'csv':
        formatter_cls = CSVTableFormatter
    elif name == 'html':
        formatter_cls = HTMLTableFormatter
    else:
        raise RuntimeError('Unknown format %s' % name)
    return formatter_cls()