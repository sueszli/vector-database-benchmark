from .utils import tilde_encode, path_with_format, HASH_LENGTH, PrefixedUrlString
import urllib

class Urls:

    def __init__(self, ds):
        if False:
            print('Hello World!')
        self.ds = ds

    def path(self, path, format=None):
        if False:
            print('Hello World!')
        if not isinstance(path, PrefixedUrlString):
            if path.startswith('/'):
                path = path[1:]
            path = self.ds.setting('base_url') + path
        if format is not None:
            path = path_with_format(path=path, format=format)
        return PrefixedUrlString(path)

    def instance(self, format=None):
        if False:
            for i in range(10):
                print('nop')
        return self.path('', format=format)

    def static(self, path):
        if False:
            print('Hello World!')
        return self.path(f'-/static/{path}')

    def static_plugins(self, plugin, path):
        if False:
            print('Hello World!')
        return self.path(f'-/static-plugins/{plugin}/{path}')

    def logout(self):
        if False:
            for i in range(10):
                print('nop')
        return self.path('-/logout')

    def database(self, database, format=None):
        if False:
            for i in range(10):
                print('nop')
        db = self.ds.get_database(database)
        return self.path(tilde_encode(db.route), format=format)

    def table(self, database, table, format=None):
        if False:
            i = 10
            return i + 15
        path = f'{self.database(database)}/{tilde_encode(table)}'
        if format is not None:
            path = path_with_format(path=path, format=format)
        return PrefixedUrlString(path)

    def query(self, database, query, format=None):
        if False:
            i = 10
            return i + 15
        path = f'{self.database(database)}/{tilde_encode(query)}'
        if format is not None:
            path = path_with_format(path=path, format=format)
        return PrefixedUrlString(path)

    def row(self, database, table, row_path, format=None):
        if False:
            return 10
        path = f'{self.table(database, table)}/{row_path}'
        if format is not None:
            path = path_with_format(path=path, format=format)
        return PrefixedUrlString(path)

    def row_blob(self, database, table, row_path, column):
        if False:
            while True:
                i = 10
        return self.table(database, table) + '/{}.blob?_blob_column={}'.format(row_path, urllib.parse.quote_plus(column))