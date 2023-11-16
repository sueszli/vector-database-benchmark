import time
from threading import RLock
from ..managers.event_manager import UpdateEvent
from ..utils import format, purge
from ..utils.struct.lock import lock
status_map = {'finished': 0, 'offline': 1, 'online': 2, 'queued': 3, 'skipped': 4, 'waiting': 5, 'temp. offline': 6, 'starting': 7, 'failed': 8, 'aborted': 9, 'decrypting': 10, 'custom': 11, 'downloading': 12, 'processing': 13, 'unknown': 14}

def _set_size(self, value):
    if False:
        i = 10
        return i + 15
    self._size = int(value)

def _set_name(self, value):
    if False:
        while True:
            i = 10
    self._name = purge.name(value, sep='')

class PyFile:
    """
    Represents a file object at runtime.
    """

    def __init__(self, manager, id, url, name, size, status, error, pluginname, package, order):
        if False:
            for i in range(10):
                print('nop')
        self.m = self.manager = manager
        self.m.cache[int(id)] = self
        self.id = int(id)
        self.url = url
        self._name = None
        self.name = name
        self._size = None
        self.size = size
        self.status = status
        self.pluginname = pluginname
        self.packageid = package
        self.error = error
        self.order = order
        self.lock = RLock()
        self.plugin = None
        self.wait_until = 0
        self.active = False
        self.abort = False
        self.reconnected = False
        self.statusname = None
        self.progress = 0
        self.maxprogress = 100
    size = property(lambda self: self._size, _set_size)
    name = property(lambda self: self._name, _set_name)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'PyFile {self.id}: {self.name}@{self.pluginname}'

    @lock
    def init_plugin(self):
        if False:
            i = 10
            return i + 15
        '\n        inits plugin instance.\n        '
        if not self.plugin:
            self.pluginmodule = self.m.pyload.plugin_manager.get_plugin(self.pluginname)
            self.pluginclass = getattr(self.pluginmodule, self.m.pyload.plugin_manager.get_plugin_name(self.pluginname))
            self.plugin = self.pluginclass(self)

    @lock
    def has_plugin(self):
        if False:
            return 10
        '\n        Thread safe way to determine this file has initialized plugin attribute.\n\n        :return:\n        '
        return hasattr(self, 'plugin') and self.plugin

    def package(self):
        if False:
            while True:
                i = 10
        '\n        return package instance.\n        '
        return self.m.get_package(self.packageid)

    def set_status(self, status):
        if False:
            print('Hello World!')
        self.status = status_map[status]
        self.sync()

    def set_custom_status(self, msg, status='processing'):
        if False:
            while True:
                i = 10
        self.statusname = msg
        self.set_status(status)

    def get_status_name(self):
        if False:
            for i in range(10):
                print('nop')
        if self.status not in (13, 14) or not self.statusname:
            return self.m.status_msg[self.status]
        else:
            return self.statusname

    def has_status(self, status):
        if False:
            i = 10
            return i + 15
        return status_map[status] == self.status

    def sync(self):
        if False:
            while True:
                i = 10
        '\n        sync PyFile instance with database.\n        '
        self.m.update_link(self)

    @lock
    def release(self):
        if False:
            print('Hello World!')
        '\n        sync and remove from cache.\n        '
        if self.packageid > 0:
            self.sync()
        if hasattr(self, 'plugin') and self.plugin:
            self.plugin.clean()
            del self.plugin
        self.m.release_link(self.id)

    def delete(self):
        if False:
            i = 10
            return i + 15
        '\n        delete pyfile from database.\n        '
        self.m.delete_link(self.id)

    def to_dict(self):
        if False:
            print('Hello World!')
        '\n        return dict with all information for interface.\n        '
        return self.to_db_dict()

    def to_db_dict(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        return data as dict for databse.\n\n        format:\n\n        {\n            id: {'url': url, 'name': name ... }\n        }\n        "
        return {self.id: {'id': self.id, 'url': self.url, 'name': self.name, 'plugin': self.pluginname, 'size': self.get_size(), 'format_size': self.format_size(), 'status': self.status, 'statusmsg': self.get_status_name(), 'package': self.packageid, 'error': self.error, 'order': self.order}}

    def abort_download(self):
        if False:
            while True:
                i = 10
        '\n        abort pyfile if possible.\n        '
        while self.id in self.m.pyload.thread_manager.processing_ids():
            self.abort = True
            if self.plugin and self.plugin.req:
                self.plugin.req.abort_downloads()
            time.sleep(0.1)
        self.abort = False
        if self.has_plugin() and self.plugin.req:
            self.plugin.req.abort_downloads()
        self.release()

    def finish_if_done(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        set status to finish and release file if every thread is finished with it.\n        '
        if self.id in self.m.pyload.thread_manager.processing_ids():
            return False
        self.set_status('finished')
        self.release()
        self.m.check_all_links_finished()
        return True

    def check_if_processed(self):
        if False:
            print('Hello World!')
        self.m.check_all_links_processed(self.id)

    def format_wait(self):
        if False:
            i = 10
            return i + 15
        '\n        formats and return wait time in human readable format.\n        '
        seconds = int(self.wait_until - time.time())
        return format.time(seconds, literally=False)

    def format_size(self):
        if False:
            while True:
                i = 10
        '\n        formats size to readable format.\n        '
        return format.size(self.get_size())

    def format_eta(self):
        if False:
            i = 10
            return i + 15
        '\n        formats eta to readable format.\n        '
        seconds = self.get_eta()
        return format.time(seconds, literally=False)

    def get_speed(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        calculates speed.\n        '
        try:
            return self.plugin.req.speed
        except Exception:
            return 0

    def get_eta(self):
        if False:
            print('Hello World!')
        '\n        gets established time of arrival.\n        '
        try:
            return int(self.get_bytes_left() // self.get_speed())
        except ZeroDivisionError:
            return 0

    def get_bytes_left(self):
        if False:
            i = 10
            return i + 15
        '\n        gets bytes left.\n        '
        try:
            return max(self.get_size() - self.plugin.req.arrived, 0)
        except Exception:
            return 0

    def get_percent(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        get % of download.\n        '
        if self.status == 12:
            try:
                return self.plugin.req.percent
            except Exception:
                return 0
        else:
            return self.progress

    def get_size(self):
        if False:
            i = 10
            return i + 15
        '\n        get size of download.\n        '
        try:
            if self.plugin.req.size:
                return self.plugin.req.size
            else:
                return self.size
        except Exception:
            return self.size

    def notify_change(self):
        if False:
            while True:
                i = 10
        e = UpdateEvent('file', self.id, 'collector' if not self.package().queue else 'queue')
        self.m.pyload.event_manager.add_event(e)

    def set_progress(self, value):
        if False:
            i = 10
            return i + 15
        if value != self.progress:
            self.progress = value
            self.notify_change()

    def set_name(self, value):
        if False:
            print('Hello World!')
        if value != self.name:
            self.name = value
            self.notify_change()