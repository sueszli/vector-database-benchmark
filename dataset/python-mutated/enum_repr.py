from enum import Enum

def repr_for_enum(enum):
    if False:
        return 10
    return '%s.%s' % (type(enum).__name__, enum.name)