from enum import IntEnum

class ATNType(IntEnum):
    LEXER = 0
    PARSER = 1

    @classmethod
    def fromOrdinal(cls, i: int):
        if False:
            return 10
        return cls._value2member_map_[i]