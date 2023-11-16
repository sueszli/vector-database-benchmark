import io
import csv
import warnings
ESCAPECHAR = '\\'

def issue12178(escapechar=ESCAPECHAR):
    if False:
        for i in range(10):
            print('nop')
    with io.StringIO(newline='') as stream:
        csv.writer(stream, escapechar=escapechar).writerow([escapechar])
        line = stream.getvalue()
    return escapechar * 2 not in line

def issue31590(line='spam%s\neggs,spam\r\n' % ESCAPECHAR, escapechar=ESCAPECHAR):
    if False:
        print('Hello World!')
    with io.StringIO(line, newline='') as stream:
        reader = csv.reader(stream, quoting=csv.QUOTE_NONE, escapechar=escapechar)
        row = next(reader)
    return len(row) != 2

def has_issue12178(dialect, affected=issue12178()):
    if False:
        print('Hello World!')
    return affected and dialect.escapechar and (dialect.quoting != csv.QUOTE_NONE)

def has_issue31590(dialect, affected=issue31590()):
    if False:
        return 10
    return affected and dialect.escapechar and (dialect.quoting == csv.QUOTE_NONE)

def warn_if_issue31590(reader):
    if False:
        while True:
            i = 10
    if has_issue31590(reader.dialect):
        warnings.warn('%r cannot parse embedded newlines correctly, see https://bugs.python.org/issue31590for details' % reader)