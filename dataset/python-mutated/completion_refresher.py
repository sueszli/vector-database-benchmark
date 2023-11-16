import threading
from .packages.special.main import COMMANDS
from collections import OrderedDict
from .sqlcompleter import SQLCompleter
from .sqlexecute import SQLExecute, ServerSpecies

class CompletionRefresher(object):
    refreshers = OrderedDict()

    def __init__(self):
        if False:
            return 10
        self._completer_thread = None
        self._restart_refresh = threading.Event()

    def refresh(self, executor, callbacks, completer_options=None):
        if False:
            for i in range(10):
                print('nop')
        'Creates a SQLCompleter object and populates it with the relevant\n        completion suggestions in a background thread.\n\n        executor - SQLExecute object, used to extract the credentials to connect\n                   to the database.\n        callbacks - A function or a list of functions to call after the thread\n                    has completed the refresh. The newly created completion\n                    object will be passed in as an argument to each callback.\n        completer_options - dict of options to pass to SQLCompleter.\n\n        '
        if completer_options is None:
            completer_options = {}
        if self.is_refreshing():
            self._restart_refresh.set()
            return [(None, None, None, 'Auto-completion refresh restarted.')]
        else:
            self._completer_thread = threading.Thread(target=self._bg_refresh, args=(executor, callbacks, completer_options), name='completion_refresh')
            self._completer_thread.daemon = True
            self._completer_thread.start()
            return [(None, None, None, 'Auto-completion refresh started in the background.')]

    def is_refreshing(self):
        if False:
            print('Hello World!')
        return self._completer_thread and self._completer_thread.is_alive()

    def _bg_refresh(self, sqlexecute, callbacks, completer_options):
        if False:
            i = 10
            return i + 15
        completer = SQLCompleter(**completer_options)
        e = sqlexecute
        executor = SQLExecute(e.dbname, e.user, e.password, e.host, e.port, e.socket, e.charset, e.local_infile, e.ssl, e.ssh_user, e.ssh_host, e.ssh_port, e.ssh_password, e.ssh_key_filename)
        if callable(callbacks):
            callbacks = [callbacks]
        while 1:
            for refresher in self.refreshers.values():
                refresher(completer, executor)
                if self._restart_refresh.is_set():
                    self._restart_refresh.clear()
                    break
            else:
                break
            continue
        for callback in callbacks:
            callback(completer)

def refresher(name, refreshers=CompletionRefresher.refreshers):
    if False:
        print('Hello World!')
    'Decorator to add the decorated function to the dictionary of\n    refreshers. Any function decorated with a @refresher will be executed as\n    part of the completion refresh routine.'

    def wrapper(wrapped):
        if False:
            i = 10
            return i + 15
        refreshers[name] = wrapped
        return wrapped
    return wrapper

@refresher('databases')
def refresh_databases(completer, executor):
    if False:
        while True:
            i = 10
    completer.extend_database_names(executor.databases())

@refresher('schemata')
def refresh_schemata(completer, executor):
    if False:
        return 10
    completer.extend_schemata(executor.dbname)
    completer.set_dbname(executor.dbname)

@refresher('tables')
def refresh_tables(completer, executor):
    if False:
        while True:
            i = 10
    completer.extend_relations(executor.tables(), kind='tables')
    completer.extend_columns(executor.table_columns(), kind='tables')

@refresher('users')
def refresh_users(completer, executor):
    if False:
        while True:
            i = 10
    completer.extend_users(executor.users())

@refresher('functions')
def refresh_functions(completer, executor):
    if False:
        for i in range(10):
            print('nop')
    completer.extend_functions(executor.functions())
    if executor.server_info.species == ServerSpecies.TiDB:
        completer.extend_functions(completer.tidb_functions, builtin=True)

@refresher('special_commands')
def refresh_special(completer, executor):
    if False:
        while True:
            i = 10
    completer.extend_special_commands(COMMANDS.keys())

@refresher('show_commands')
def refresh_show_commands(completer, executor):
    if False:
        print('Hello World!')
    completer.extend_show_items(executor.show_candidates())

@refresher('keywords')
def refresh_keywords(completer, executor):
    if False:
        print('Hello World!')
    if executor.server_info.species == ServerSpecies.TiDB:
        completer.extend_keywords(completer.tidb_keywords, replace=True)