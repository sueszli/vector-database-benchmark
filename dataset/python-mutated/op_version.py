from ..base import core
__all__ = []

def Singleton(cls):
    if False:
        while True:
            i = 10
    _instance = {}

    def _singleton(*args, **kargs):
        if False:
            print('Hello World!')
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]
    return _singleton

class OpUpdateInfoHelper:

    def __init__(self, info):
        if False:
            while True:
                i = 10
        self._info = info

    def verify_key_value(self, name=''):
        if False:
            print('Hello World!')
        result = False
        key_funcs = {core.OpAttrInfo: 'name', core.OpInputOutputInfo: 'name'}
        if name == '':
            result = True
        elif type(self._info) in key_funcs:
            if getattr(self._info, key_funcs[type(self._info)])() == name:
                result = True
        return result

@Singleton
class OpLastCheckpointChecker:

    def __init__(self):
        if False:
            return 10
        self.raw_version_map = core.get_op_version_map()
        self.checkpoints_map = {}
        self._construct_map()

    def _construct_map(self):
        if False:
            print('Hello World!')
        for op_name in self.raw_version_map:
            last_checkpoint = self.raw_version_map[op_name].checkpoints()[-1]
            infos = last_checkpoint.version_desc().infos()
            self.checkpoints_map[op_name] = infos

    def filter_updates(self, op_name, type=core.OpUpdateType.kInvalid, key=''):
        if False:
            return 10
        updates = []
        if op_name in self.checkpoints_map:
            for update in self.checkpoints_map[op_name]:
                if update.type() == type or type == core.OpUpdateType.kInvalid:
                    if OpUpdateInfoHelper(update.info()).verify_key_value(key):
                        updates.append(update.info())
        return updates