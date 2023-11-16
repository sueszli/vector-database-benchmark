import re
import sqlparse

class DelimiterCommand(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self._delimiter = ';'

    def _split(self, sql):
        if False:
            return 10
        'Temporary workaround until sqlparse.split() learns about custom\n        delimiters.'
        placeholder = '￼'
        if self._delimiter == ';':
            return sqlparse.split(sql)
        while placeholder in sql:
            placeholder += placeholder[0]
        sql = sql.replace(';', placeholder)
        sql = sql.replace(self._delimiter, ';')
        split = sqlparse.split(sql)
        return [stmt.replace(';', self._delimiter).replace(placeholder, ';') for stmt in split]

    def queries_iter(self, input):
        if False:
            return 10
        'Iterate over queries in the input string.'
        queries = self._split(input)
        while queries:
            for sql in queries:
                delimiter = self._delimiter
                sql = queries.pop(0)
                if sql.endswith(delimiter):
                    trailing_delimiter = True
                    sql = sql.strip(delimiter)
                else:
                    trailing_delimiter = False
                yield sql
                if self._delimiter != delimiter:
                    combined_statement = ' '.join([sql] + queries)
                    if trailing_delimiter:
                        combined_statement += delimiter
                    queries = self._split(combined_statement)[1:]

    def set(self, arg, **_):
        if False:
            print('Hello World!')
        'Change delimiter.\n\n        Since `arg` is everything that follows the DELIMITER token\n        after sqlparse (it may include other statements separated by\n        the new delimiter), we want to set the delimiter to the first\n        word of it.\n\n        '
        match = arg and re.search('[^\\s]+', arg)
        if not match:
            message = 'Missing required argument, delimiter'
            return [(None, None, None, message)]
        delimiter = match.group()
        if delimiter.lower() == 'delimiter':
            return [(None, None, None, 'Invalid delimiter "delimiter"')]
        self._delimiter = delimiter
        return [(None, None, None, 'Changed delimiter to {}'.format(delimiter))]

    @property
    def current(self):
        if False:
            while True:
                i = 10
        return self._delimiter