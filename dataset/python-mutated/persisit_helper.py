import hashlib
from collections import OrderedDict
from rqalpha.const import PERSIST_MODE
from rqalpha.core.events import EVENT
from rqalpha.utils.logger import system_log

class PersistHelper(object):

    def __init__(self, persist_provider, event_bus, persist_mode):
        if False:
            while True:
                i = 10
        self._objects = OrderedDict()
        self._last_state = {}
        self._persist_provider = persist_provider
        if persist_mode == PERSIST_MODE.REAL_TIME:
            event_bus.add_listener(EVENT.POST_BEFORE_TRADING, self.persist)
            event_bus.add_listener(EVENT.POST_AFTER_TRADING, self.persist)
            event_bus.add_listener(EVENT.POST_BAR, self.persist)
            event_bus.add_listener(EVENT.DO_PERSIST, self.persist)
            event_bus.add_listener(EVENT.POST_SETTLEMENT, self.persist)
            event_bus.add_listener(EVENT.DO_RESTORE, self.restore)

    def persist(self, *_):
        if False:
            for i in range(10):
                print('nop')
        for (key, obj) in self._objects.items():
            try:
                state = obj.get_state()
                if not state:
                    continue
                md5 = hashlib.md5(state).hexdigest()
                if self._last_state.get(key) == md5:
                    continue
                self._persist_provider.store(key, state)
            except Exception as e:
                system_log.exception('PersistHelper.persist fail')
            else:
                self._last_state[key] = md5

    def register(self, key, obj):
        if False:
            print('Hello World!')
        if key in self._objects:
            raise RuntimeError('duplicated persist key found: {}'.format(key))
        self._objects[key] = obj

    def unregister(self, key):
        if False:
            while True:
                i = 10
        if key in self._objects:
            del self._objects[key]
            return True
        return False

    def restore(self, event):
        if False:
            return 10
        key = getattr(event, 'key', None)
        if key:
            return self._restore_obj(key, self._objects[key])
        ret = {key: self._restore_obj(key, obj) for (key, obj) in self._objects.items()}
        return ret

    def _restore_obj(self, key, obj):
        if False:
            while True:
                i = 10
        state = self._persist_provider.load(key)
        system_log.debug('restore {} with state = {}', key, state)
        if not state:
            return False
        try:
            obj.set_state(state)
        except Exception:
            system_log.exception('restore failed: key={} state={}'.format(key, state))
        return True