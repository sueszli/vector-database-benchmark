from salt.modules.inspectlib.entities import AllowedDir, IgnoredDir, Package, PackageCfgFile, PayloadFile
from salt.modules.inspectlib.fsdb import CsvDB

class DBHandleBase:
    """
    Handle for the *volatile* database, which serves the purpose of caching
    the inspected data. This database can be destroyed or corrupted, so it should
    be simply re-created from scratch.
    """

    def __init__(self, path):
        if False:
            for i in range(10):
                print('nop')
        '\n        Constructor.\n        '
        self._path = path
        self.init_queries = list()
        self._db = CsvDB(self._path)

    def open(self, new=False):
        if False:
            return 10
        '\n        Init the database, if required.\n        '
        self._db.new() if new else self._db.open()
        self._run_init_queries()

    def _run_init_queries(self):
        if False:
            return 10
        '\n        Initialization queries\n        '
        for obj in (Package, PackageCfgFile, PayloadFile, IgnoredDir, AllowedDir):
            self._db.create_table_from_object(obj())

    def purge(self):
        if False:
            i = 10
            return i + 15
        '\n        Purge whole database.\n        '
        for table_name in self._db.list_tables():
            self._db.flush(table_name)
        self._run_init_queries()

    def flush(self, table):
        if False:
            i = 10
            return i + 15
        '\n        Flush the table.\n        '
        self._db.flush(table)

    def close(self):
        if False:
            i = 10
            return i + 15
        '\n        Close the database connection.\n        '
        self._db.close()

    def __getattr__(self, item):
        if False:
            for i in range(10):
                print('nop')
        '\n        Proxy methods from the Database instance.\n\n        :param item:\n        :return:\n        '
        return getattr(self._db, item)

class DBHandle(DBHandleBase):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Keep singleton.\n        '
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, path):
        if False:
            i = 10
            return i + 15
        '\n        Database handle for the specific\n\n        :param path:\n        :return:\n        '
        DBHandleBase.__init__(self, path)