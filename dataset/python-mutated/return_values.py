class ObjectWithSetItemCap:

    def __init__(self) -> None:
        if False:
            return 10
        self._dict = {}

    def clear(self):
        if False:
            print('Hello World!')
        self._dict = {}

    def __setitem__(self, item, value):
        if False:
            for i in range(10):
                print('nop')
        self._dict[item] = value

    def __getitem__(self, item):
        if False:
            for i in range(10):
                print('nop')
        return self._dict[item]

    @property
    def container(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dict

class ObjectWithoutSetItemCap:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        pass
OBJECT_WITH_SETITEM_CAP = ObjectWithSetItemCap()
OBJECT_WITHOUT_SETITEM_CAP = ObjectWithoutSetItemCap()