from enum import Enum

class AutoName(Enum):

    def _generate_next_value_(self, *args):
        if False:
            return 10
        return self.lower()

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'pyrogram.enums.{self}'