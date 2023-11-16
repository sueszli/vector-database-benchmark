from ..managers.event_manager import UpdateEvent
from ..utils.old import safepath

class PyPackage:
    """
    Represents a package object at runtime.
    """

    def __init__(self, manager, id, name, folder, site, password, queue, order):
        if False:
            while True:
                i = 10
        self.m = self.manager = manager
        self.m.package_cache[int(id)] = self
        self.id = int(id)
        self.name = name
        self._folder = folder
        self.site = site
        self.password = password
        self.queue = queue
        self.order = order
        self.set_finished = False

    @property
    def folder(self):
        if False:
            while True:
                i = 10
        return safepath(self._folder)

    def to_dict(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a dictionary representation of the data.\n\n        :return: dict: {id: { attr: value }}\n        '
        return {self.id: {'id': self.id, 'name': self.name, 'folder': self.folder, 'site': self.site, 'password': self.password, 'queue': self.queue, 'order': self.order, 'links': {}}}

    def get_children(self):
        if False:
            while True:
                i = 10
        '\n        get information about contained links.\n        '
        return self.m.get_package_data(self.id)['links']

    def sync(self):
        if False:
            return 10
        '\n        sync with db.\n        '
        self.m.update_package(self)

    def release(self):
        if False:
            print('Hello World!')
        '\n        sync and delete from cache.\n        '
        self.sync()
        self.m.release_package(self.id)

    def delete(self):
        if False:
            i = 10
            return i + 15
        self.m.delete_package(self.id)

    def notify_change(self):
        if False:
            return 10
        e = UpdateEvent('pack', self.id, 'collector' if not self.queue else 'queue')
        self.m.pyload.event_manager.add_event(e)