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

class LuxSQLTable(lux.LuxDataFrame):
    """
    A subclass of Lux.LuxDataFrame that houses other variables and functions for generating visual recommendations. Does not support normal pandas functionality.
    """
    _metadata = ['_intent', '_inferred_intent', '_data_type', 'unique_values', 'cardinality', '_rec_info', '_min_max', '_current_vis', '_widget', '_recommendation', '_prev', '_history', '_saved_export', '_sampled', '_toggle_pandas_display', '_message', '_pandas_only', 'pre_aggregated', '_type_override', '_length', '_setup_done']

    def __init__(self, *args, table_name='', **kw):
        if False:
            print('Hello World!')
        super(LuxSQLTable, self).__init__(*args, **kw)
        if lux.config.executor.name != 'GeneralDatabaseExecutor':
            from lux.executor.SQLExecutor import SQLExecutor
            lux.config.executor = SQLExecutor()
        self._length = 0
        self._setup_done = False
        if table_name != '':
            self.set_SQL_table(table_name)
        warnings.formatwarning = lux.warning_format

    def __len__(self):
        if False:
            return 10
        if self._setup_done:
            return self._length
        else:
            return super(LuxSQLTable, self).__len__()

    def set_SQL_table(self, t_name):
        if False:
            print('Hello World!')
        if self.table_name != '':
            warnings.warn(f"\nThis LuxSQLTable is already tied to a database table. Please create a new Lux dataframe and connect it to your table '{t_name}'.", stacklevel=2)
        else:
            self.table_name = t_name
        try:
            lux.config.executor.compute_dataset_metadata(self)
        except Exception as error:
            error_str = str(error)
            if f'relation "{t_name}" does not exist' in error_str:
                warnings.warn(f"\nThe table '{t_name}' does not exist in your database./", stacklevel=2)

    def maintain_metadata(self):
        if False:
            while True:
                i = 10
        if not hasattr(self, '_metadata_fresh') or not self._metadata_fresh:
            lux.config.executor.compute_dataset_metadata(self)
            self._infer_structure()
            self._metadata_fresh = True

    def expire_metadata(self):
        if False:
            while True:
                i = 10
        '\n        Expire all saved metadata to trigger a recomputation the next time the data is required.\n        '

    def _ipython_display_(self):
        if False:
            print('Hello World!')
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
                    while True:
                        i = 10
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