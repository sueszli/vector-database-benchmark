from enum import Enum

class ToolConfigKeyType(Enum):
    STRING = 'string'
    FILE = 'file'
    INT = 'int'

    @classmethod
    def get_key_type(cls, store):
        if False:
            while True:
                i = 10
        store = store.upper()
        if store in cls.__members__:
            return cls[store]
        raise ValueError(f'{store} is not a valid key type.')

    def __str__(self):
        if False:
            return 10
        return self.value