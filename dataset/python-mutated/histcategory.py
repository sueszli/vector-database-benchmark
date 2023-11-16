"""A completion category that queries the SQL history store."""
from typing import Optional
from qutebrowser.qt.sql import QSqlQueryModel
from qutebrowser.qt.widgets import QWidget
from qutebrowser.misc import sql
from qutebrowser.utils import debug, message, log
from qutebrowser.config import config
from qutebrowser.completion.models import util

class HistoryCategory(QSqlQueryModel):
    """A completion category that queries the SQL history store."""

    def __init__(self, *, database: sql.Database, delete_func: util.DeleteFuncType=None, parent: QWidget=None) -> None:
        if False:
            while True:
                i = 10
        'Create a new History completion category.'
        super().__init__(parent=parent)
        self._database = database
        self.name = 'History'
        self._query: Optional[sql.Query] = None
        self.columns_to_filter = [0, 1]
        self.delete_func = delete_func
        self._empty_prefix: Optional[str] = None

    def _atime_expr(self):
        if False:
            print('Hello World!')
        'If max_items is set, return an expression to limit the query.'
        max_items = config.val.completion.web_history.max_items
        assert max_items != 0
        if max_items < 0:
            return ''
        min_atime = self._database.query(' '.join(['SELECT min(last_atime) FROM', '(SELECT last_atime FROM CompletionHistory', 'ORDER BY last_atime DESC LIMIT :limit)'])).run(limit=max_items).value()
        if not min_atime:
            return ''
        return 'AND last_atime >= {}'.format(min_atime)

    def set_pattern(self, pattern):
        if False:
            print('Hello World!')
        'Set the pattern used to filter results.\n\n        Args:\n            pattern: string pattern to filter by.\n        '
        raw_pattern = pattern
        if self._empty_prefix is not None and raw_pattern.startswith(self._empty_prefix):
            log.sql.debug('Skipping query on {} due to prefix {} returning nothing.'.format(raw_pattern, self._empty_prefix))
            return
        self._empty_prefix = None
        pattern = pattern.replace('%', '\\%')
        pattern = pattern.replace('_', '\\_')
        words = ['%{}%'.format(w) for w in pattern.split(' ')]
        where_clause = ' AND '.join(("(url LIKE :{val} escape '\\' OR title LIKE :{val} escape '\\')".format(val=i) for i in range(len(words))))
        timestamp_format = config.val.completion.timestamp_format or ''
        timefmt = "strftime('{}', last_atime, 'unixepoch', 'localtime')".format(timestamp_format.replace("'", '`'))
        try:
            if not self._query or len(words) != len(self._query.bound_values()):
                self._query = self._database.query(' '.join(['SELECT url, title, {}'.format(timefmt), 'FROM CompletionHistory', 'WHERE ({})'.format(where_clause), self._atime_expr(), 'ORDER BY last_atime DESC']), forward_only=False)
            with debug.log_time('sql', 'Running completion query'):
                self._query.run(**{str(i): w for (i, w) in enumerate(words)})
        except sql.KnownError as e:
            message.error('Error with SQL query: {}'.format(e.text()))
            return
        self.setQuery(self._query.query)
        if not self.rowCount() and (not self.canFetchMore()):
            self._empty_prefix = raw_pattern

    def removeRows(self, row, _count, _parent=None):
        if False:
            while True:
                i = 10
        'Override QAbstractItemModel::removeRows to re-run SQL query.'
        assert self._query is not None
        with debug.log_time('sql', 'Re-running completion query post-delete'):
            self._query.run()
        self.setQuery(self._query.query)
        while self.rowCount() < row:
            self.fetchMore()
        return True