from collections import Counter
from picard.config import get_config
from picard.util import LockableObject

class DataObject(LockableObject):

    def __init__(self, obj_id):
        if False:
            return 10
        super().__init__()
        self.id = obj_id
        self.genres = Counter()
        self.item = None

    def add_genre(self, name, count):
        if False:
            while True:
                i = 10
        if name:
            self.genres[name] += count

    @staticmethod
    def set_genre_inc_params(inc, config=None):
        if False:
            for i in range(10):
                print('nop')
        require_authentication = False
        config = config or get_config()
        if config.setting['use_genres']:
            use_folksonomy = config.setting['folksonomy_tags']
            if config.setting['only_my_genres']:
                require_authentication = True
                inc |= {'user-tags'} if use_folksonomy else {'user-genres'}
            else:
                inc |= {'tags'} if use_folksonomy else {'genres'}
        return require_authentication