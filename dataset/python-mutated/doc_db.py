"""Documents, in a sqlite database."""
import sqlite3
from . import utils
from . import DEFAULTS

class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None):
        if False:
            return 10
        self.path = db_path or DEFAULTS['db_path']
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self.close()

    def path(self):
        if False:
            print('Hello World!')
        'Return the path to the file that backs this database.'
        return self.path

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        'Close the connection to the database.'
        self.connection.close()

    def get_doc_ids(self):
        if False:
            return 10
        'Fetch all ids of docs stored in the db.'
        cursor = self.connection.cursor()
        cursor.execute('SELECT id FROM documents')
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text(self, doc_id):
        if False:
            return 10
        "Fetch the raw text of the doc for 'doc_id'."
        cursor = self.connection.cursor()
        cursor.execute('SELECT text FROM documents WHERE id = ?', (utils.normalize(doc_id),))
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]