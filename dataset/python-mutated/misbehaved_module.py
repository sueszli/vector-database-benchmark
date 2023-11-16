import types

class _MisbehavedModule(types.ModuleType):

    @property
    def __spec__(self):
        if False:
            for i in range(10):
                print('nop')
        raise Exception('Oh noes!')
MisbehavedModule = _MisbehavedModule('MisbehavedModule')