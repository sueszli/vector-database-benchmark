import pandas as pd
from lux.core.series import LuxSeries
from lux.vis.Clause import Clause
from lux.vis.Vis import Vis
from lux.vis.VisList import VisList
from lux.history.history import History
from lux.utils.date_utils import is_datetime_series
from lux.utils.message import Message
from lux.utils.utils import check_import_lux_widget
from typing import Dict, Union, List, Callable
import warnings
import traceback
import lux

class JoinedSQLTable(lux.LuxSQLTable):
    """
    A subclass of Lux.LuxDataFrame that houses other variables and functions for generating visual recommendations. Does not support normal pandas functionality.
    """
    _metadata = ['_intent', '_inferred_intent', '_data_type', 'unique_values', 'cardinality', '_rec_info', '_min_max', '_current_vis', '_widget', '_recommendation', '_prev', '_history', '_saved_export', '_sampled', '_toggle_pandas_display', '_message', '_pandas_only', 'pre_aggregated', '_type_override', 'joins', 'using_view']

    def __init__(self, *args, joins=[], **kw):
        if False:
            return 10
        super(JoinedSQLTable, self).__init__(*args, **kw)
        from lux.executor.SQLExecutor import SQLExecutor
        lux.config.executor = SQLExecutor()
        tables = self.extract_tables(joins)
        if len(tables) > 4:
            warnings.warn(f'\nPlease provide a maximum of 4 (Four) unique tables to ensure optimal performance.', stacklevel=2)
        view_name = self.create_view(tables, joins)
        self._length = 0
        if view_name != '':
            self.set_SQL_table(view_name)
        warnings.formatwarning = lux.warning_format

    def len(self):
        if False:
            while True:
                i = 10
        return self._length

    def extract_tables(self, joins):
        if False:
            return 10
        tables = set()
        for condition in joins:
            lhs = condition[0:condition.index('=')].strip()
            rhs = condition[condition.index('=') + 1:].strip()
            table1 = lhs[0:lhs.index('.')].strip()
            table2 = rhs[0:rhs.index('.')].strip()
            tables.add(table1)
            tables.add(table2)
        return tables

    def create_view(self, tables, joins):
        if False:
            i = 10
            return i + 15
        import psycopg2
        dbc = lux.config.SQLconnection.cursor()
        import time
        curr_time = str(int(time.time()))
        viewname = 'lux_view_' + curr_time
        table_entry = ''
        for (idx, table) in enumerate(tables, 1):
            table_entry += table
            if idx < len(tables):
                table_entry += ', '
        condition_entry = ''
        for (idx, join) in enumerate(joins, 1):
            condition_entry += join
            if idx < len(joins):
                condition_entry += ' AND '
        try:
            s = 'CREATE VIEW {} AS SELECT * FROM {} where {}'.format(viewname, table_entry, condition_entry)
            dbc.execute(s)
            lux.config.SQLconnection.commit()
        except Exception as error:
            print('Exception : ' + str(error))
            viewname = ''
        dbc.close()
        return viewname

    def _ipython_display_(self):
        if False:
            while True:
                i = 10
        from IPython.display import HTML, Markdown, display
        from IPython.display import clear_output
        import ipywidgets as widgets
        try:
            if self._pandas_only:
                display(self.display_pandas())
                self._pandas_only = False
            if not self.index.nlevels >= 2 or self.columns.nlevels >= 2:
                self.maintain_metadata()
                if self._intent != [] and (not hasattr(self, '_compiled') or not self._compiled):
                    from lux.processor.Compiler import Compiler
                    self.current_vis = Compiler.compile_intent(self, self._intent)
            if lux.config.default_display == 'lux':
                self._toggle_pandas_display = False
            else:
                self._toggle_pandas_display = True
            self.maintain_recs()
            self._widget.observe(self.remove_deleted_recs, names='deletedIndices')
            self._widget.observe(self.set_intent_on_click, names='selectedIntentIndex')
            button = widgets.Button(description='Toggle Table/Lux', layout=widgets.Layout(width='200px', top='6px', bottom='6px'))
            self.output = widgets.Output()
            self._sampled = lux.config.executor.execute_preview(self)
            display(button, self.output)

            def on_button_clicked(b):
                if False:
                    return 10
                with self.output:
                    if b:
                        self._toggle_pandas_display = not self._toggle_pandas_display
                    clear_output()
                    connect_str = self.table_name
                    connection_type = str(type(lux.config.SQLconnection))
                    if 'psycopg2.extensions.connection' in connection_type:
                        connection_dsn = lux.config.SQLconnection.get_dsn_parameters()
                        host_name = connection_dsn['host']
                        host_port = connection_dsn['port']
                        dbname = connection_dsn['dbname']
                        connect_str = host_name + ':' + host_port + '/' + dbname
                    elif 'sqlalchemy.engine.base.Engine' in connection_type:
                        db_connection = str(lux.config.SQLconnection)
                        db_start = db_connection.index('@') + 1
                        db_end = len(db_connection) - 1
                        connect_str = db_connection[db_start:db_end]
                    if self._toggle_pandas_display:
                        notification = 'Here is a preview of the **{}** database table: **{}**'.format(self.table_name, connect_str)
                        display(Markdown(notification), self._sampled.display_pandas())
                    else:
                        display(self._widget)
            button.on_click(on_button_clicked)
            on_button_clicked(None)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            if lux.config.pandas_fallback:
                warnings.warn('\nUnexpected error in rendering Lux widget and recommendations. Falling back to Pandas display.\nPlease report the following issue on Github: https://github.com/lux-org/lux/issues \n', stacklevel=2)
                warnings.warn(traceback.format_exc())
                display(self.display_pandas())
            else:
                raise

    def head(self, n: int=5):
        if False:
            return 10
        return

    def tail(self, n: int=5):
        if False:
            i = 10
            return i + 15
        return

    def info(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return

    def describe(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return

    def groupby(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return