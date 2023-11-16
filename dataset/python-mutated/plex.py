from __future__ import unicode_literals
from future.builtins import object
from future.builtins import str
import plexpy
if plexpy.PYTHON2:
    import logger
else:
    from plexpy import logger

class DummyObject(object):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        pass

    def __call__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __getattr__(self, attr):
        if False:
            while True:
                i = 10
        return self

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __next__(self):
        if False:
            return 10
        raise StopIteration
PlexObject = DummyObject

def initialize_plexapi():
    if False:
        for i in range(10):
            print('nop')
    from plexapi.server import PlexServer
    global PlexObject
    PlexObject = PlexServer

class Plex(object):

    def __init__(self, url=None, token=None):
        if False:
            print('Hello World!')
        url = url or plexpy.CONFIG.PMS_URL
        token = token or plexpy.CONFIG.PMS_TOKEN
        self.PlexServer = PlexObject(url, token)

    def get_library(self, section_id):
        if False:
            for i in range(10):
                print('nop')
        return self.PlexServer.library.sectionByID(int(section_id))

    def get_library_items(self, section_id):
        if False:
            print('Hello World!')
        return self.get_library(section_id).all()

    def get_item(self, rating_key):
        if False:
            while True:
                i = 10
        return self.PlexServer.fetchItem(rating_key)