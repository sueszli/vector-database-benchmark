from volatility.renderers.basic import Renderer, Bytes
from volatility import debug
import sqlite3

class SqliteRenderer(Renderer):

    def __init__(self, plugin_name, config):
        if False:
            i = 10
            return i + 15
        self._plugin_name = plugin_name
        self._config = config
        self._db = None
        self._accumulator = [0, []]
    column_types = [(str, 'TEXT'), (int, 'TEXT'), (float, 'TEXT'), (Bytes, 'BLOB')]

    def _column_type(self, col_type):
        if False:
            i = 10
            return i + 15
        for (t, v) in self.column_types:
            if issubclass(col_type, t):
                return v
        return 'TEXT'

    def _sanitize_name(self, name):
        if False:
            while True:
                i = 10
        return name

    def render(self, outfd, grid):
        if False:
            i = 10
            return i + 15
        if not self._config.OUTPUT_FILE:
            debug.error('Please specify a valid output file using --output-file')
        self._db = sqlite3.connect(self._config.OUTPUT_FILE, isolation_level=None)
        self._db.text_factory = str
        create = 'CREATE TABLE IF NOT EXISTS ' + self._plugin_name + '( id INTEGER, ' + ', '.join(['"' + self._sanitize_name(i.name) + '" ' + self._column_type(i.type) for i in grid.columns]) + ')'
        self._db.execute(create)

        def _add_multiple_row(node, accumulator):
            if False:
                for i in range(10):
                    print('nop')
            accumulator[0] = accumulator[0] + 1
            accumulator[1].append([accumulator[0]] + [str(v) for v in node.values])
            if len(accumulator[1]) > 20000:
                self._db.execute('BEGIN TRANSACTION')
                insert = 'INSERT INTO ' + self._plugin_name + ' (id, ' + ', '.join(['"' + self._sanitize_name(i.name) + '"' for i in grid.columns]) + ') ' + ' VALUES (?, ' + ', '.join(['?'] * len(node.values)) + ')'
                self._db.executemany(insert, accumulator[1])
                accumulator = [accumulator[0], []]
                self._db.execute('COMMIT TRANSACTION')
            self._accumulator = accumulator
            return accumulator
        grid.populate(_add_multiple_row, self._accumulator)
        if len(self._accumulator[1]) > 0:
            self._db.execute('BEGIN TRANSACTION')
            insert = 'INSERT INTO ' + self._plugin_name + ' (id, ' + ', '.join(['"' + self._sanitize_name(i.name) + '"' for i in grid.columns]) + ') ' + ' VALUES (?, ' + ', '.join(['?'] * (len(self._accumulator[1][0]) - 1)) + ')'
            self._db.executemany(insert, self._accumulator[1])
            self._db.execute('COMMIT TRANSACTION')