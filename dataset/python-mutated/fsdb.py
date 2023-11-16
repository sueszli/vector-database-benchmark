"""
    :codeauthor: Bo Maryniuk <bo@suse.de>
"""
import csv
import datetime
import gzip
import os
import re
import shutil
import sys
from salt.utils.odict import OrderedDict

class CsvDBEntity:
    """
    Serializable object for the table.
    """

    def _serialize(self, description):
        if False:
            print('Hello World!')
        '\n        Serialize the object to a row for CSV according to the table description.\n\n        :return:\n        '
        return [getattr(self, attr) for attr in description]

class CsvDB:
    """
    File-based CSV database.
    This database is in-memory operating relatively small plain text csv files.
    """

    def __init__(self, path):
        if False:
            i = 10
            return i + 15
        '\n        Constructor to store the database files.\n\n        :param path:\n        '
        self._prepare(path)
        self._opened = False
        self.db_path = None
        self._opened = False
        self._tables = {}

    def _prepare(self, path):
        if False:
            i = 10
            return i + 15
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def _label(self):
        if False:
            i = 10
            return i + 15
        '\n        Create label of the database, based on the date-time.\n\n        :return:\n        '
        return datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')

    def new(self):
        if False:
            i = 10
            return i + 15
        '\n        Create a new database and opens it.\n\n        :return:\n        '
        dbname = self._label()
        self.db_path = os.path.join(self.path, dbname)
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
        self._opened = True
        self.list_tables()
        return dbname

    def purge(self, dbid):
        if False:
            print('Hello World!')
        '\n        Purge the database.\n\n        :param dbid:\n        :return:\n        '
        db_path = os.path.join(self.path, dbid)
        if os.path.exists(db_path):
            shutil.rmtree(db_path, ignore_errors=True)
            return True
        return False

    def flush(self, table):
        if False:
            for i in range(10):
                print('nop')
        '\n        Flush table.\n\n        :param table:\n        :return:\n        '
        table_path = os.path.join(self.db_path, table)
        if os.path.exists(table_path):
            os.unlink(table_path)

    def list(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        List all the databases on the given path.\n\n        :return:\n        '
        databases = []
        for dbname in os.listdir(self.path):
            databases.append(dbname)
        return list(reversed(sorted(databases)))

    def list_tables(self):
        if False:
            print('Hello World!')
        '\n        Load existing tables and their descriptions.\n\n        :return:\n        '
        if not self._tables:
            for table_name in os.listdir(self.db_path):
                self._tables[table_name] = self._load_table(table_name)
        return self._tables.keys()

    def _load_table(self, table_name):
        if False:
            for i in range(10):
                print('nop')
        with gzip.open(os.path.join(self.db_path, table_name), 'rt') as table:
            return OrderedDict([tuple(elm.split(':')) for elm in next(csv.reader(table))])

    def open(self, dbname=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Open database from the path with the name or latest.\n        If there are no yet databases, create a new implicitly.\n\n        :return:\n        '
        databases = self.list()
        if self.is_closed():
            self.db_path = os.path.join(self.path, dbname or (databases and databases[0] or self.new()))
            if not self._opened:
                self.list_tables()
                self._opened = True

    def close(self):
        if False:
            i = 10
            return i + 15
        '\n        Close the database.\n\n        :return:\n        '
        self._opened = False

    def is_closed(self):
        if False:
            i = 10
            return i + 15
        '\n        Return if the database is closed.\n\n        :return:\n        '
        return not self._opened

    def create_table_from_object(self, obj):
        if False:
            return 10
        "\n        Create a table from the object.\n        NOTE: This method doesn't stores anything.\n\n        :param obj:\n        :return:\n        "
        get_type = lambda item: str(type(item)).split("'")[1]
        if not os.path.exists(os.path.join(self.db_path, obj._TABLE)):
            with gzip.open(os.path.join(self.db_path, obj._TABLE), 'wt') as table_file:
                csv.writer(table_file).writerow(['{col}:{type}'.format(col=elm[0], type=get_type(elm[1])) for elm in tuple(obj.__dict__.items())])
            self._tables[obj._TABLE] = self._load_table(obj._TABLE)

    def store(self, obj, distinct=False):
        if False:
            return 10
        '\n        Store an object in the table.\n\n        :param obj: An object to store\n        :param distinct: Store object only if there is none identical of such.\n                          If at least one field is different, store it.\n        :return:\n        '
        if distinct:
            fields = dict(zip(self._tables[obj._TABLE].keys(), obj._serialize(self._tables[obj._TABLE])))
            db_obj = self.get(obj.__class__, eq=fields)
            if db_obj and distinct:
                raise Exception('Object already in the database.')
        with gzip.open(os.path.join(self.db_path, obj._TABLE), 'at') as table:
            csv.writer(table).writerow(self._validate_object(obj))

    def update(self, obj, matches=None, mt=None, lt=None, eq=None):
        if False:
            print('Hello World!')
        '\n        Update object(s) in the database.\n\n        :param obj:\n        :param matches:\n        :param mt:\n        :param lt:\n        :param eq:\n        :return:\n        '
        updated = False
        objects = list()
        for _obj in self.get(obj.__class__):
            if self.__criteria(_obj, matches=matches, mt=mt, lt=lt, eq=eq):
                objects.append(obj)
                updated = True
            else:
                objects.append(_obj)
        self.flush(obj._TABLE)
        self.create_table_from_object(obj)
        for obj in objects:
            self.store(obj)
        return updated

    def delete(self, obj, matches=None, mt=None, lt=None, eq=None):
        if False:
            return 10
        '\n        Delete object from the database.\n\n        :param obj:\n        :param matches:\n        :param mt:\n        :param lt:\n        :param eq:\n        :return:\n        '
        deleted = False
        objects = list()
        for _obj in self.get(obj):
            if not self.__criteria(_obj, matches=matches, mt=mt, lt=lt, eq=eq):
                objects.append(_obj)
            else:
                deleted = True
        self.flush(obj._TABLE)
        self.create_table_from_object(obj())
        for _obj in objects:
            self.store(_obj)
        return deleted

    def _validate_object(self, obj):
        if False:
            while True:
                i = 10
        descr = self._tables.get(obj._TABLE)
        if descr is None:
            raise Exception('Table {} not found.'.format(obj._TABLE))
        return obj._serialize(self._tables[obj._TABLE])

    def __criteria(self, obj, matches=None, mt=None, lt=None, eq=None):
        if False:
            while True:
                i = 10
        '\n        Returns True if object is aligned to the criteria.\n\n        :param obj:\n        :param matches:\n        :param mt:\n        :param lt:\n        :param eq:\n        :return: Boolean\n        '
        for (field, value) in (mt or {}).items():
            if getattr(obj, field) <= value:
                return False
        for (field, value) in (lt or {}).items():
            if getattr(obj, field) >= value:
                return False
        for (field, value) in (eq or {}).items():
            if getattr(obj, field) != value:
                return False
        for (field, value) in (matches or {}).items():
            if not re.search(value, str(getattr(obj, field))):
                return False
        return True

    def get(self, obj, matches=None, mt=None, lt=None, eq=None):
        if False:
            i = 10
            return i + 15
        '\n        Get objects from the table.\n\n        :param table_name:\n        :param matches: Regexp.\n        :param mt: More than.\n        :param lt: Less than.\n        :param eq: Equals.\n        :return:\n        '
        objects = []
        with gzip.open(os.path.join(self.db_path, obj._TABLE), 'rt') as table:
            header = None
            for data in csv.reader(table):
                if not header:
                    header = data
                    continue
                _obj = obj()
                for (t_attr, t_data) in zip(header, data):
                    (t_attr, t_type) = t_attr.split(':')
                    setattr(_obj, t_attr, self._to_type(t_data, t_type))
                if self.__criteria(_obj, matches=matches, mt=mt, lt=lt, eq=eq):
                    objects.append(_obj)
        return objects

    def _to_type(self, data, type):
        if False:
            return 10
        if type == 'int':
            data = int(data)
        elif type == 'float':
            data = float(data)
        elif type == 'long':
            data = sys.version_info[0] == 2 and long(data) or int(data)
        else:
            data = str(data)
        return data