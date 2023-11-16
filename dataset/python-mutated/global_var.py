import pickle
from rqalpha.utils.logger import user_system_log, system_log

class GlobalVars(object):

    def get_state(self):
        if False:
            return 10
        dict_data = {}
        for (key, value) in self.__dict__.items():
            try:
                dict_data[key] = pickle.dumps(value)
            except Exception:
                user_system_log.warn('g.{} can not pickle', key)
        return pickle.dumps(dict_data)

    def set_state(self, state):
        if False:
            for i in range(10):
                print('nop')
        dict_data = pickle.loads(state)
        for (key, value) in dict_data.items():
            try:
                self.__dict__[key] = pickle.loads(value)
                system_log.debug('restore g.{} {}', key, type(self.__dict__[key]))
            except Exception:
                user_system_log.warn('g.{} restore failed', key)