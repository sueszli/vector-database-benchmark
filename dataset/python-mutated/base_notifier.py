from typing import Optional

class BaseNotifier(object):

    def __init__(self, _id: str):
        if False:
            return 10
        self._id = _id

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'<{self.__class__.__name__} object at {id(self)}>'

    def notify(self, message: Optional[str]=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError