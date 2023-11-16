import os
import re
import shutil
import json
import time
import errno
from collections import defaultdict
import sqlite3
import gevent.event
import util
from util import SafeRe
from Db.Db import Db
from Debug import Debug
from Config import config
from util import helper
from util import ThreadPool
from Plugin import PluginManager
from Translate import translate as _
thread_pool_fs_read = ThreadPool.ThreadPool(config.threads_fs_read, name='FS read')
thread_pool_fs_write = ThreadPool.ThreadPool(config.threads_fs_write, name='FS write')
thread_pool_fs_batch = ThreadPool.ThreadPool(1, name='FS batch')

@PluginManager.acceptPlugins
class SiteStorage(object):

    def __init__(self, site, allow_create=True):
        if False:
            print('Hello World!')
        self.site = site
        self.directory = '%s/%s' % (config.data_dir, self.site.address)
        self.allowed_dir = os.path.abspath(self.directory)
        self.log = site.log
        self.db = None
        self.db_checked = False
        self.event_db_busy = None
        self.has_db = self.isFile('dbschema.json')
        if not os.path.isdir(self.directory):
            if allow_create:
                os.mkdir(self.directory)
            else:
                raise Exception('Directory not exists: %s' % self.directory)

    def getDbFile(self):
        if False:
            print('Hello World!')
        if self.db:
            return self.db.schema['db_file']
        elif self.isFile('dbschema.json'):
            schema = self.loadJson('dbschema.json')
            return schema['db_file']
        else:
            return False

    def openDb(self, close_idle=False):
        if False:
            print('Hello World!')
        schema = self.getDbSchema()
        db_path = self.getPath(schema['db_file'])
        return Db(schema, db_path, close_idle=close_idle)

    def closeDb(self, reason='Unknown (SiteStorage)'):
        if False:
            return 10
        if self.db:
            self.db.close(reason)
        self.event_db_busy = None
        self.db = None

    def getDbSchema(self):
        if False:
            while True:
                i = 10
        try:
            self.site.needFile('dbschema.json')
            schema = self.loadJson('dbschema.json')
        except Exception as err:
            raise Exception('dbschema.json is not a valid JSON: %s' % err)
        return schema

    def loadDb(self):
        if False:
            while True:
                i = 10
        self.log.debug('No database, waiting for dbschema.json...')
        self.site.needFile('dbschema.json', priority=3)
        self.log.debug('Got dbschema.json')
        self.has_db = self.isFile('dbschema.json')
        if self.has_db:
            schema = self.getDbSchema()
            db_path = self.getPath(schema['db_file'])
            if not os.path.isfile(db_path) or os.path.getsize(db_path) == 0:
                try:
                    self.rebuildDb(reason='Missing database')
                except Exception as err:
                    self.log.error(err)
                    pass
            if self.db:
                self.db.close('Gettig new db for SiteStorage')
            self.db = self.openDb(close_idle=True)
            try:
                changed_tables = self.db.checkTables()
                if changed_tables:
                    self.rebuildDb(delete_db=False, reason='Changed tables')
            except sqlite3.OperationalError:
                pass

    @util.Noparallel()
    def getDb(self):
        if False:
            return 10
        if self.event_db_busy:
            self.log.debug('Wating for db...')
            self.event_db_busy.get()
        if not self.db:
            self.loadDb()
        return self.db

    def updateDbFile(self, inner_path, file=None, cur=None):
        if False:
            while True:
                i = 10
        path = self.getPath(inner_path)
        if cur:
            db = cur.db
        else:
            db = self.getDb()
        return db.updateJson(path, file, cur)

    @thread_pool_fs_read.wrap
    def getDbFiles(self):
        if False:
            print('Hello World!')
        found = 0
        for (content_inner_path, content) in self.site.content_manager.contents.items():
            if self.isFile(content_inner_path):
                yield (content_inner_path, self.getPath(content_inner_path))
            else:
                self.log.debug('[MISSING] %s' % content_inner_path)
            content_inner_path_dir = helper.getDirname(content_inner_path)
            for file_relative_path in list(content.get('files', {}).keys()) + list(content.get('files_optional', {}).keys()):
                if not file_relative_path.endswith('.json') and (not file_relative_path.endswith('json.gz')):
                    continue
                file_inner_path = content_inner_path_dir + file_relative_path
                file_inner_path = file_inner_path.strip('/')
                if self.isFile(file_inner_path):
                    yield (file_inner_path, self.getPath(file_inner_path))
                else:
                    self.log.debug('[MISSING] %s' % file_inner_path)
                found += 1
                if found % 100 == 0:
                    time.sleep(0.001)

    @util.Noparallel()
    @thread_pool_fs_batch.wrap
    def rebuildDb(self, delete_db=True, reason='Unknown'):
        if False:
            for i in range(10):
                print('nop')
        self.log.info('Rebuilding db (reason: %s)...' % reason)
        self.has_db = self.isFile('dbschema.json')
        if not self.has_db:
            return False
        schema = self.loadJson('dbschema.json')
        db_path = self.getPath(schema['db_file'])
        if os.path.isfile(db_path) and delete_db:
            if self.db:
                self.closeDb('rebuilding')
                time.sleep(0.5)
            self.log.info('Deleting %s' % db_path)
            try:
                os.unlink(db_path)
            except Exception as err:
                self.log.error('Delete error: %s' % err)
        if not self.db:
            self.db = self.openDb()
        self.event_db_busy = gevent.event.AsyncResult()
        self.log.info('Rebuild: Creating tables...')
        self.db.checkTables()
        cur = self.db.getCursor()
        cur.logging = False
        s = time.time()
        self.log.info('Rebuild: Getting db files...')
        db_files = list(self.getDbFiles())
        num_imported = 0
        num_total = len(db_files)
        num_error = 0
        self.log.info('Rebuild: Importing data...')
        try:
            if num_total > 100:
                self.site.messageWebsocket(_['Database rebuilding...<br>Imported {0} of {1} files (error: {2})...'].format('0000', num_total, num_error), 'rebuild', 0)
            for (file_inner_path, file_path) in db_files:
                try:
                    if self.updateDbFile(file_inner_path, file=open(file_path, 'rb'), cur=cur):
                        num_imported += 1
                except Exception as err:
                    self.log.error('Error importing %s: %s' % (file_inner_path, Debug.formatException(err)))
                    num_error += 1
                if num_imported and num_imported % 100 == 0:
                    self.site.messageWebsocket(_['Database rebuilding...<br>Imported {0} of {1} files (error: {2})...'].format(num_imported, num_total, num_error), 'rebuild', int(float(num_imported) / num_total * 100))
                    time.sleep(0.001)
        finally:
            cur.close()
            if num_total > 100:
                self.site.messageWebsocket(_['Database rebuilding...<br>Imported {0} of {1} files (error: {2})...'].format(num_imported, num_total, num_error), 'rebuild', 100)
            self.log.info('Rebuild: Imported %s data file in %.3fs' % (num_imported, time.time() - s))
            self.event_db_busy.set(True)
            self.event_db_busy = None
            self.db.commit('Rebuilt')
        return True

    def query(self, query, params=None):
        if False:
            print('Hello World!')
        if not query.strip().upper().startswith('SELECT'):
            raise Exception('Only SELECT query supported')
        try:
            res = self.getDb().execute(query, params)
        except sqlite3.DatabaseError as err:
            if err.__class__.__name__ == 'DatabaseError':
                self.log.error('Database error: %s, query: %s, try to rebuilding it...' % (err, query))
                try:
                    self.rebuildDb(reason='Query error')
                except sqlite3.OperationalError:
                    pass
                res = self.db.cur.execute(query, params)
            else:
                raise err
        return res

    def ensureDir(self, inner_path):
        if False:
            print('Hello World!')
        try:
            os.makedirs(self.getPath(inner_path))
        except OSError as err:
            if err.errno == errno.EEXIST:
                return False
            else:
                raise err
        return True

    def open(self, inner_path, mode='rb', create_dirs=False, **kwargs):
        if False:
            i = 10
            return i + 15
        file_path = self.getPath(inner_path)
        if create_dirs:
            file_inner_dir = os.path.dirname(inner_path)
            self.ensureDir(file_inner_dir)
        return open(file_path, mode, **kwargs)

    @thread_pool_fs_read.wrap
    def read(self, inner_path, mode='rb'):
        if False:
            i = 10
            return i + 15
        return open(self.getPath(inner_path), mode).read()

    @thread_pool_fs_write.wrap
    def writeThread(self, inner_path, content):
        if False:
            print('Hello World!')
        file_path = self.getPath(inner_path)
        self.ensureDir(os.path.dirname(inner_path))
        if hasattr(content, 'read'):
            with open(file_path, 'wb') as file:
                shutil.copyfileobj(content, file)
        elif inner_path == 'content.json' and os.path.isfile(file_path):
            helper.atomicWrite(file_path, content)
        else:
            with open(file_path, 'wb') as file:
                file.write(content)

    def write(self, inner_path, content):
        if False:
            print('Hello World!')
        self.writeThread(inner_path, content)
        self.onUpdated(inner_path)

    def delete(self, inner_path):
        if False:
            i = 10
            return i + 15
        file_path = self.getPath(inner_path)
        os.unlink(file_path)
        self.onUpdated(inner_path, file=False)

    def deleteDir(self, inner_path):
        if False:
            return 10
        dir_path = self.getPath(inner_path)
        os.rmdir(dir_path)

    def rename(self, inner_path_before, inner_path_after):
        if False:
            return 10
        for retry in range(3):
            rename_err = None
            try:
                os.rename(self.getPath(inner_path_before), self.getPath(inner_path_after))
                break
            except Exception as err:
                rename_err = err
                self.log.error('%s rename error: %s (retry #%s)' % (inner_path_before, err, retry))
                time.sleep(0.1 + retry)
        if rename_err:
            raise rename_err

    @thread_pool_fs_read.wrap
    def walk(self, dir_inner_path, ignore=None):
        if False:
            while True:
                i = 10
        directory = self.getPath(dir_inner_path)
        for (root, dirs, files) in os.walk(directory):
            root = root.replace('\\', '/')
            root_relative_path = re.sub('^%s' % re.escape(directory), '', root).lstrip('/')
            for file_name in files:
                if root_relative_path:
                    file_relative_path = root_relative_path + '/' + file_name
                else:
                    file_relative_path = file_name
                if ignore and SafeRe.match(ignore, file_relative_path):
                    continue
                yield file_relative_path
            if ignore:
                dirs_filtered = []
                for dir_name in dirs:
                    if root_relative_path:
                        dir_relative_path = root_relative_path + '/' + dir_name
                    else:
                        dir_relative_path = dir_name
                    if ignore == '.*' or re.match('.*([|(]|^)%s([|)]|$)' % re.escape(dir_relative_path + '/.*'), ignore):
                        continue
                    dirs_filtered.append(dir_name)
                dirs[:] = dirs_filtered

    @thread_pool_fs_read.wrap
    def list(self, dir_inner_path):
        if False:
            while True:
                i = 10
        directory = self.getPath(dir_inner_path)
        return os.listdir(directory)

    def onUpdated(self, inner_path, file=None):
        if False:
            while True:
                i = 10
        should_load_to_db = inner_path.endswith('.json') or inner_path.endswith('.json.gz')
        if inner_path == 'dbschema.json':
            self.has_db = self.isFile('dbschema.json')
            if self.has_db:
                self.closeDb('New dbschema')
                gevent.spawn(self.getDb)
        elif not config.disable_db and should_load_to_db and self.has_db:
            if config.verbose:
                self.log.debug('Loading json file to db: %s (file: %s)' % (inner_path, file))
            try:
                self.updateDbFile(inner_path, file)
            except Exception as err:
                self.log.error('Json %s load error: %s' % (inner_path, Debug.formatException(err)))
                self.closeDb('Json load error')

    @thread_pool_fs_read.wrap
    def loadJson(self, inner_path):
        if False:
            for i in range(10):
                print('nop')
        with self.open(inner_path, 'r', encoding='utf8') as file:
            return json.load(file)

    def writeJson(self, inner_path, data):
        if False:
            return 10
        self.write(inner_path, helper.jsonDumps(data).encode('utf8'))

    def getSize(self, inner_path):
        if False:
            return 10
        path = self.getPath(inner_path)
        try:
            return os.path.getsize(path)
        except Exception:
            return 0

    def isFile(self, inner_path):
        if False:
            print('Hello World!')
        return os.path.isfile(self.getPath(inner_path))

    def isExists(self, inner_path):
        if False:
            i = 10
            return i + 15
        return os.path.exists(self.getPath(inner_path))

    def isDir(self, inner_path):
        if False:
            while True:
                i = 10
        return os.path.isdir(self.getPath(inner_path))

    def getPath(self, inner_path):
        if False:
            i = 10
            return i + 15
        inner_path = inner_path.replace('\\', '/')
        if not inner_path:
            return self.directory
        if '../' in inner_path:
            raise Exception('File not allowed: %s' % inner_path)
        return '%s/%s' % (self.directory, inner_path)

    def getInnerPath(self, path):
        if False:
            i = 10
            return i + 15
        if path == self.directory:
            inner_path = ''
        elif path.startswith(self.directory):
            inner_path = path[len(self.directory) + 1:]
        else:
            raise Exception('File not allowed: %s' % path)
        return inner_path

    def verifyFiles(self, quick_check=False, add_optional=False, add_changed=True):
        if False:
            return 10
        bad_files = []
        back = defaultdict(int)
        back['bad_files'] = bad_files
        i = 0
        self.log.debug('Verifing files...')
        if not self.site.content_manager.contents.get('content.json'):
            self.log.debug('VerifyFile content.json not exists')
            self.site.needFile('content.json', update=True)
            self.site.content_manager.loadContent()
        for (content_inner_path, content) in list(self.site.content_manager.contents.items()):
            back['num_content'] += 1
            i += 1
            if i % 50 == 0:
                time.sleep(0.001)
            if not os.path.isfile(self.getPath(content_inner_path)):
                back['num_content_missing'] += 1
                self.log.debug('[MISSING] %s' % content_inner_path)
                bad_files.append(content_inner_path)
            for file_relative_path in list(content.get('files', {}).keys()):
                back['num_file'] += 1
                file_inner_path = helper.getDirname(content_inner_path) + file_relative_path
                file_inner_path = file_inner_path.strip('/')
                file_path = self.getPath(file_inner_path)
                if not os.path.isfile(file_path):
                    back['num_file_missing'] += 1
                    self.log.debug('[MISSING] %s' % file_inner_path)
                    bad_files.append(file_inner_path)
                    continue
                if quick_check:
                    ok = os.path.getsize(file_path) == content['files'][file_relative_path]['size']
                    if not ok:
                        err = 'Invalid size'
                else:
                    try:
                        ok = self.site.content_manager.verifyFile(file_inner_path, open(file_path, 'rb'))
                    except Exception as err:
                        ok = False
                if not ok:
                    back['num_file_invalid'] += 1
                    self.log.debug('[INVALID] %s: %s' % (file_inner_path, err))
                    if add_changed or content.get('cert_user_id'):
                        bad_files.append(file_inner_path)
            optional_added = 0
            optional_removed = 0
            for file_relative_path in list(content.get('files_optional', {}).keys()):
                back['num_optional'] += 1
                file_node = content['files_optional'][file_relative_path]
                file_inner_path = helper.getDirname(content_inner_path) + file_relative_path
                file_inner_path = file_inner_path.strip('/')
                file_path = self.getPath(file_inner_path)
                hash_id = self.site.content_manager.hashfield.getHashId(file_node['sha512'])
                if not os.path.isfile(file_path):
                    if self.site.content_manager.isDownloaded(file_inner_path, hash_id):
                        back['num_optional_removed'] += 1
                        self.log.debug('[OPTIONAL MISSING] %s' % file_inner_path)
                        self.site.content_manager.optionalRemoved(file_inner_path, hash_id, file_node['size'])
                    if add_optional and self.site.isDownloadable(file_inner_path):
                        self.log.debug('[OPTIONAL ADDING] %s' % file_inner_path)
                        bad_files.append(file_inner_path)
                    continue
                if quick_check:
                    ok = os.path.getsize(file_path) == content['files_optional'][file_relative_path]['size']
                else:
                    try:
                        ok = self.site.content_manager.verifyFile(file_inner_path, open(file_path, 'rb'))
                    except Exception as err:
                        ok = False
                if ok:
                    if not self.site.content_manager.isDownloaded(file_inner_path, hash_id):
                        back['num_optional_added'] += 1
                        self.site.content_manager.optionalDownloaded(file_inner_path, hash_id, file_node['size'])
                        optional_added += 1
                        self.log.debug('[OPTIONAL FOUND] %s' % file_inner_path)
                else:
                    if self.site.content_manager.isDownloaded(file_inner_path, hash_id):
                        back['num_optional_removed'] += 1
                        self.site.content_manager.optionalRemoved(file_inner_path, hash_id, file_node['size'])
                        optional_removed += 1
                    bad_files.append(file_inner_path)
                    self.log.debug('[OPTIONAL CHANGED] %s' % file_inner_path)
            if config.verbose:
                self.log.debug('%s verified: %s, quick: %s, optionals: +%s -%s' % (content_inner_path, len(content['files']), quick_check, optional_added, optional_removed))
        self.site.content_manager.contents.db.processDelayed()
        time.sleep(0.001)
        return back

    def updateBadFiles(self, quick_check=True):
        if False:
            return 10
        s = time.time()
        res = self.verifyFiles(quick_check, add_optional=True, add_changed=not self.site.settings.get('own'))
        bad_files = res['bad_files']
        self.site.bad_files = {}
        if bad_files:
            for bad_file in bad_files:
                self.site.bad_files[bad_file] = 1
        self.log.debug('Checked files in %.2fs... Found bad files: %s, Quick:%s' % (time.time() - s, len(bad_files), quick_check))

    @thread_pool_fs_batch.wrap
    def deleteFiles(self):
        if False:
            while True:
                i = 10
        site_title = self.site.content_manager.contents.get('content.json', {}).get('title', self.site.address)
        message_id = 'delete-%s' % self.site.address
        self.log.debug('Deleting files from content.json (title: %s)...' % site_title)
        files = []
        content_inner_paths = list(self.site.content_manager.contents.keys())
        for (i, content_inner_path) in enumerate(content_inner_paths):
            content = self.site.content_manager.contents.get(content_inner_path, {})
            files.append(content_inner_path)
            for file_relative_path in list(content.get('files', {}).keys()):
                file_inner_path = helper.getDirname(content_inner_path) + file_relative_path
                files.append(file_inner_path)
            for file_relative_path in list(content.get('files_optional', {}).keys()):
                file_inner_path = helper.getDirname(content_inner_path) + file_relative_path
                files.append(file_inner_path)
            if i % 100 == 0:
                num_files = len(files)
                self.site.messageWebsocket(_('Deleting site <b>{site_title}</b>...<br>Collected {num_files} files'), message_id, i / len(content_inner_paths) * 25)
        if self.isFile('dbschema.json'):
            self.log.debug('Deleting db file...')
            self.closeDb('Deleting site')
            self.has_db = False
            try:
                schema = self.loadJson('dbschema.json')
                db_path = self.getPath(schema['db_file'])
                if os.path.isfile(db_path):
                    os.unlink(db_path)
            except Exception as err:
                self.log.error('Db file delete error: %s' % err)
        num_files = len(files)
        for (i, inner_path) in enumerate(files):
            path = self.getPath(inner_path)
            if os.path.isfile(path):
                for retry in range(5):
                    try:
                        os.unlink(path)
                        break
                    except Exception as err:
                        self.log.error('Error removing %s: %s, try #%s' % (inner_path, err, retry))
                    time.sleep(float(retry) / 10)
            if i % 100 == 0:
                self.site.messageWebsocket(_('Deleting site <b>{site_title}</b>...<br>Deleting file {i}/{num_files}'), message_id, 25 + i / num_files * 50)
            self.onUpdated(inner_path, False)
        self.log.debug('Deleting empty dirs...')
        i = 0
        for (root, dirs, files) in os.walk(self.directory, topdown=False):
            for dir in dirs:
                path = os.path.join(root, dir)
                if os.path.isdir(path):
                    try:
                        i += 1
                        if i % 100 == 0:
                            self.site.messageWebsocket(_('Deleting site <b>{site_title}</b>...<br>Deleting empty directories {i}'), message_id, 85)
                        os.rmdir(path)
                    except OSError:
                        pass
        if os.path.isdir(self.directory) and os.listdir(self.directory) == []:
            os.rmdir(self.directory)
        if os.path.isdir(self.directory):
            self.log.debug('Some unknown file remained in site data dir: %s...' % self.directory)
            self.site.messageWebsocket(_('Deleting site <b>{site_title}</b>...<br>Site deleted, but some unknown files left in the directory'), message_id, 100)
            return False
        else:
            self.log.debug('Site %s data directory deleted: %s...' % (site_title, self.directory))
            self.site.messageWebsocket(_('Deleting site <b>{site_title}</b>...<br>All files deleted successfully'), message_id, 100)
            return True