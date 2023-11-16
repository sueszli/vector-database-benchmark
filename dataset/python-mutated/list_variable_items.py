def get_variables():
    if False:
        i = 10
        return i + 15
    return {'MIXED USAGE': MixedUsage()}

class MixedUsage:

    def __init__(self):
        if False:
            print('Hello World!')
        self.data = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

    def __getitem__(self, item):
        if False:
            i = 10
            return i + 15
        if isinstance(item, slice) and item.start is item.stop is item.step is None:
            return self
        if isinstance(item, (int, slice)):
            return self.data[item]
        if isinstance(item, str):
            return self.data.index(item)
        raise TypeError