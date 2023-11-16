import apsw
import os
from contextlib import suppress
from threading import Lock
from time import time_ns
from calibre.constants import cache_dir
creation_sql = '\nCREATE TABLE IF NOT EXISTS last_read_positions ( id INTEGER PRIMARY KEY AUTOINCREMENT,\n    library_id TEXT NOT NULL,\n    book INTEGER NOT NULL,\n    format TEXT NOT NULL COLLATE NOCASE,\n    user TEXT NOT NULL,\n    cfi TEXT NOT NULL,\n    epoch INTEGER NOT NULL,\n    pos_frac REAL NOT NULL DEFAULT 0,\n    tooltip TEXT NOT NULL,\n    UNIQUE(user, library_id, book, format)\n);\nCREATE INDEX IF NOT EXISTS users_id ON last_read_positions (user);\n'
lock = Lock()

class LastReadCache:

    def __init__(self, path='', limit=5):
        if False:
            print('Hello World!')
        self.limit = limit
        self.conn = apsw.Connection(path or os.path.join(cache_dir(), 'srv-last-read.sqlite'))
        self.execute(creation_sql)

    def get(self, *args, **kw):
        if False:
            i = 10
            return i + 15
        ans = self.conn.cursor().execute(*args)
        if kw.get('all', True):
            return ans.fetchall()
        with suppress(StopIteration, IndexError):
            return next(ans)[0]

    def execute(self, sql, bindings=None):
        if False:
            for i in range(10):
                print('nop')
        cursor = self.conn.cursor()
        return cursor.execute(sql, bindings)

    def add_last_read_position(self, library_id, book_id, fmt, user, cfi, pos_frac, tooltip):
        if False:
            return 10
        with lock, self.conn:
            if not cfi:
                self.execute('DELETE FROM last_read_positions WHERE library_id=? AND book=? AND format=? AND user=?', (library_id, book_id, fmt, user))
            else:
                epoch = time_ns()
                self.execute('INSERT OR REPLACE INTO last_read_positions(library_id,book,format,user,cfi,epoch,pos_frac,tooltip) VALUES (?,?,?,?,?,?,?,?)', (library_id, book_id, fmt, user, cfi, epoch, pos_frac, tooltip))
                items = tuple(self.get('SELECT id FROM last_read_positions WHERE user=? ORDER BY id DESC', (user,), all=True))
                if len(items) > self.limit:
                    self.execute('DELETE FROM last_read_positions WHERE user=? AND id <= ?', (user, items[self.limit][0]))
                return epoch

    def get_recently_read(self, user):
        if False:
            return 10
        with lock:
            ans = []
            for (library_id, book, fmt, cfi, epoch, pos_frac, tooltip) in self.execute('SELECT library_id,book,format,cfi,epoch,pos_frac,tooltip FROM last_read_positions WHERE user=? ORDER BY epoch DESC', (user,)):
                ans.append({'library_id': library_id, 'book_id': book, 'format': fmt, 'cfi': cfi, 'epoch': epoch, 'pos_frac': pos_frac, 'tooltip': tooltip})
            return ans
path_cache = {}

def last_read_cache(path=''):
    if False:
        i = 10
        return i + 15
    with lock:
        ans = path_cache.get(path)
        if ans is None:
            ans = path_cache[path] = LastReadCache(path)
    return ans