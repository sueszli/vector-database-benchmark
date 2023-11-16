import errno
import json
import os
import tempfile
import time
from hashlib import sha1
from itertools import count
from calibre import walk
from calibre.constants import cache_dir, iswindows
from calibre.ptempfile import TemporaryFile
from calibre.srv.render_book import RENDER_VERSION
from calibre.utils.filenames import rmtree
from calibre.utils.ipc.simple_worker import start_pipe_worker
from calibre.utils.lock import ExclusiveFile
from calibre.utils.serialize import msgpack_dumps
from calibre.utils.short_uuid import uuid4
from polyglot.builtins import as_bytes, as_unicode, iteritems
DAY = 24 * 3600
VIEWER_VERSION = 1
td_counter = count()

def book_cache_dir():
    if False:
        i = 10
        return i + 15
    return getattr(book_cache_dir, 'override', os.path.join(cache_dir(), 'ev2'))

def cache_lock():
    if False:
        for i in range(10):
            print('nop')
    return ExclusiveFile(os.path.join(book_cache_dir(), 'metadata.json'), timeout=600)

def book_hash(path, size, mtime):
    if False:
        return 10
    path = os.path.normcase(os.path.abspath(path))
    raw = json.dumps((path, size, mtime, RENDER_VERSION, VIEWER_VERSION))
    if not isinstance(raw, bytes):
        raw = raw.encode('utf-8')
    return as_unicode(sha1(raw).hexdigest())

def safe_makedirs(path):
    if False:
        return 10
    try:
        os.makedirs(path)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise
    return path

def robust_rmtree(x):
    if False:
        i = 10
        return i + 15
    retries = 2 if iswindows else 1
    for i in range(retries):
        try:
            try:
                rmtree(x)
            except UnicodeDecodeError:
                rmtree(as_bytes(x))
            return True
        except OSError:
            time.sleep(0.1)
    return False

def robust_rename(a, b):
    if False:
        for i in range(10):
            print('nop')
    retries = 20 if iswindows else 1
    for i in range(retries):
        try:
            os.rename(a, b)
            return True
        except OSError:
            time.sleep(0.1)
    return False

def clear_temp(temp_path):
    if False:
        return 10
    now = time.time()
    for x in os.listdir(temp_path):
        x = os.path.join(temp_path, x)
        mtime = os.path.getmtime(x)
        if now - mtime > DAY:
            robust_rmtree(x)

def expire_cache(path, instances, max_age):
    if False:
        return 10
    now = time.time()
    remove = [x for x in instances if now - x['atime'] > max_age and x['status'] == 'finished']
    for instance in remove:
        if robust_rmtree(os.path.join(path, instance['path'])):
            instances.remove(instance)

def expire_old_versions(path, instances):
    if False:
        for i in range(10):
            print('nop')
    instances = filter(lambda x: x['status'] == 'finished', instances)
    remove = sorted(instances, key=lambda x: x['atime'], reverse=True)[1:]
    for instance in remove:
        if robust_rmtree(os.path.join(path, instance['path'])):
            yield instance

def expire_cache_and_temp(temp_path, finished_path, metadata, max_age, force_expire):
    if False:
        return 10
    now = time.time()
    if now - metadata['last_clear_at'] < DAY and max_age >= 0 and (not force_expire):
        return
    clear_temp(temp_path)
    entries = metadata['entries']
    path_key_map = {}
    for (key, instances) in tuple(entries.items()):
        if instances:
            expire_cache(finished_path, instances, max_age)
            if not instances:
                del entries[key]
            else:
                for x in instances:
                    book_path = x.get('book_path')
                    if book_path:
                        path_key_map.setdefault(book_path, []).append(key)
    for keys in path_key_map.values():
        instances = []
        for key in keys:
            instances += entries.get(key, [])
        if len(instances) > 1:
            removed = tuple(expire_old_versions(finished_path, instances))
            if removed:
                for r in removed:
                    rkey = r['key']
                    if rkey in entries:
                        try:
                            entries[rkey].remove(r)
                        except ValueError:
                            pass
                        if not entries[rkey]:
                            del entries[rkey]
    metadata['last_clear_at'] = now

def prepare_convert(temp_path, key, st, book_path):
    if False:
        i = 10
        return i + 15
    tdir = tempfile.mkdtemp(dir=temp_path, prefix=f'c{next(td_counter)}-')
    now = time.time()
    return {'path': os.path.basename(tdir), 'id': uuid4(), 'status': 'working', 'mtime': now, 'atime': now, 'key': key, 'file_mtime': st.st_mtime, 'file_size': st.st_size, 'cache_size': 0, 'book_path': book_path}

class ConversionFailure(ValueError):

    def __init__(self, book_path, worker_output):
        if False:
            i = 10
            return i + 15
        self.book_path = book_path
        self.worker_output = worker_output
        ValueError.__init__(self, f'Failed to convert book: {book_path} with error:\n{worker_output}')
running_workers = []

def clean_running_workers():
    if False:
        i = 10
        return i + 15
    for p in running_workers:
        if p.poll() is None:
            p.kill()
    del running_workers[:]

def do_convert(path, temp_path, key, instance):
    if False:
        i = 10
        return i + 15
    tdir = os.path.join(temp_path, instance['path'])
    p = None
    try:
        with TemporaryFile('log.txt') as logpath:
            with open(logpath, 'w+b') as logf:
                p = start_pipe_worker('from calibre.srv.render_book import viewer_main; viewer_main()', stdout=logf, stderr=logf)
                running_workers.append(p)
                p.stdin.write(msgpack_dumps((path, tdir, {'size': instance['file_size'], 'mtime': instance['file_mtime'], 'hash': key})))
                p.stdin.close()
            if p.wait() != 0:
                with open(logpath, 'rb') as logf:
                    worker_output = logf.read().decode('utf-8', 'replace')
                raise ConversionFailure(path, worker_output)
    finally:
        try:
            running_workers.remove(p)
        except Exception:
            pass
    size = 0
    for f in walk(tdir):
        size += os.path.getsize(f)
    instance['cache_size'] = size

def save_metadata(metadata, f):
    if False:
        while True:
            i = 10
    (f.seek(0), f.truncate(), f.write(as_bytes(json.dumps(metadata, indent=2))))

def prepare_book(path, convert_func=do_convert, max_age=30 * DAY, force=False, prepare_notify=None, force_expire=False):
    if False:
        print('Hello World!')
    st = os.stat(path)
    key = book_hash(path, st.st_size, st.st_mtime)
    finished_path = safe_makedirs(os.path.join(book_cache_dir(), 'f'))
    temp_path = safe_makedirs(os.path.join(book_cache_dir(), 't'))
    with cache_lock() as f:
        try:
            metadata = json.loads(f.read())
        except ValueError:
            metadata = {'entries': {}, 'last_clear_at': 0}
        entries = metadata['entries']
        instances = entries.setdefault(key, [])
        for instance in tuple(instances):
            if instance['status'] == 'finished':
                if force:
                    robust_rmtree(os.path.join(finished_path, instance['path']))
                    instances.remove(instance)
                else:
                    instance['atime'] = time.time()
                    save_metadata(metadata, f)
                    return os.path.join(finished_path, instance['path'])
        if prepare_notify:
            prepare_notify()
        instance = prepare_convert(temp_path, key, st, path)
        instances.append(instance)
        save_metadata(metadata, f)
    convert_func(path, temp_path, key, instance)
    src_path = os.path.join(temp_path, instance['path'])
    with cache_lock() as f:
        ans = tempfile.mkdtemp(dir=finished_path, prefix=f'c{next(td_counter)}-')
        instance['path'] = os.path.basename(ans)
        try:
            metadata = json.loads(f.read())
        except ValueError:
            metadata = {'entries': {}, 'last_clear_at': 0}
        entries = metadata['entries']
        instances = entries.setdefault(key, [])
        os.rmdir(ans)
        if not robust_rename(src_path, ans):
            raise Exception('Failed to rename: "{}" to "{}" probably some software such as an antivirus or file sync program running on your computer has locked the files'.format(src_path, ans))
        instance['status'] = 'finished'
        for q in instances:
            if q['id'] == instance['id']:
                q.update(instance)
                break
        expire_cache_and_temp(temp_path, finished_path, metadata, max_age, force_expire)
        save_metadata(metadata, f)
    return ans

def update_book(path, old_stat, name_data_map=None):
    if False:
        print('Hello World!')
    old_key = book_hash(path, old_stat.st_size, old_stat.st_mtime)
    finished_path = safe_makedirs(os.path.join(book_cache_dir(), 'f'))
    with cache_lock() as f:
        st = os.stat(path)
        new_key = book_hash(path, st.st_size, st.st_mtime)
        if old_key == new_key:
            return
        try:
            metadata = json.loads(f.read())
        except ValueError:
            metadata = {'entries': {}, 'last_clear_at': 0}
        entries = metadata['entries']
        instances = entries.get(old_key)
        if not instances:
            return
        for instance in tuple(instances):
            if instance['status'] == 'finished':
                entries.setdefault(new_key, []).append(instance)
                instances.remove(instance)
                if not instances:
                    del entries[old_key]
                instance['file_mtime'] = st.st_mtime
                instance['file_size'] = st.st_size
                if name_data_map:
                    for (name, data) in iteritems(name_data_map):
                        with open(os.path.join(finished_path, instance['path'], name), 'wb') as f2:
                            f2.write(data)
                save_metadata(metadata, f)
                return

def find_tests():
    if False:
        return 10
    import unittest

    class TestViewerCache(unittest.TestCase):
        ae = unittest.TestCase.assertEqual

        def setUp(self):
            if False:
                print('Hello World!')
            self.tdir = tempfile.mkdtemp()
            book_cache_dir.override = os.path.join(self.tdir, 'ev2')

        def tearDown(self):
            if False:
                print('Hello World!')
            rmtree(self.tdir)
            del book_cache_dir.override

        def test_viewer_cache(self):
            if False:
                for i in range(10):
                    print('nop')

            def convert_mock(path, temp_path, key, instance):
                if False:
                    print('Hello World!')
                self.ae(instance['status'], 'working')
                self.ae(instance['key'], key)
                with open(os.path.join(temp_path, instance['path'], 'sentinel'), 'wb') as f:
                    f.write(b'test')

            def set_data(x):
                if False:
                    i = 10
                    return i + 15
                if not isinstance(x, bytes):
                    x = x.encode('utf-8')
                with open(book_src, 'wb') as f:
                    f.write(x)
            book_src = os.path.join(self.tdir, 'book.epub')
            set_data('a')
            path = prepare_book(book_src, convert_func=convert_mock)

            def read(x, mode='r'):
                if False:
                    for i in range(10):
                        print('nop')
                with open(x, mode) as f:
                    return f.read()
            self.ae(read(os.path.join(path, 'sentinel'), 'rb'), b'test')
            second_path = prepare_book(book_src, convert_func=convert_mock)
            self.ae(path, second_path)
            set_data('bc')
            third_path = prepare_book(book_src, convert_func=convert_mock)
            self.assertNotEqual(path, third_path)
            fourth_path = prepare_book(book_src, convert_func=convert_mock)
            self.ae(third_path, fourth_path)
            fourth_path = prepare_book(book_src, convert_func=convert_mock, force=True)
            self.assertNotEqual(third_path, fourth_path)
            set_data('bcd')
            prepare_book(book_src, convert_func=convert_mock, max_age=-1000)
            self.ae([], os.listdir(os.path.join(book_cache_dir(), 'f')))
            opath = prepare_book(book_src, convert_func=convert_mock, force_expire=True)
            finished_entries = os.listdir(os.path.join(book_cache_dir(), 'f'))
            self.ae(len(finished_entries), 1)
            set_data('bcde' * 4096)
            npath = prepare_book(book_src, convert_func=convert_mock, force_expire=True)
            new_finished_entries = os.listdir(os.path.join(book_cache_dir(), 'f'))
            self.ae(len(new_finished_entries), 1)
            self.assertNotEqual(opath, npath)
            set_data('bcdef')
            prepare_book(book_src, convert_func=convert_mock, max_age=-1000, force_expire=True)
            self.ae([], os.listdir(os.path.join(book_cache_dir(), 'f')))
            with cache_lock() as f:
                metadata = json.loads(f.read())
                self.assertEqual(metadata['entries'], {})
            book_src = os.path.join(self.tdir, 'book2.epub')
            set_data('bb')
            path = prepare_book(book_src, convert_func=convert_mock)
            self.ae(read(os.path.join(path, 'sentinel'), 'rb'), b'test')
            bs = os.stat(book_src)
            set_data('cde')
            update_book(book_src, bs, name_data_map={'sentinel': b'updated'})
            self.ae(read(os.path.join(path, 'sentinel'), 'rb'), b'updated')
            self.ae(1, len(os.listdir(os.path.join(book_cache_dir(), 'f'))))
            with cache_lock() as f:
                metadata = json.loads(f.read())
            self.ae(len(metadata['entries']), 1)
            entry = list(metadata['entries'].values())[0]
            self.ae(len(entry), 1)
            entry = entry[0]
            st = os.stat(book_src)
            self.ae(entry['file_size'], st.st_size)
            self.ae(entry['file_mtime'], st.st_mtime)
    return unittest.defaultTestLoader.loadTestsFromTestCase(TestViewerCache)