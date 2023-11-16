"""A shared value without deep copy."""

class SharedValue(object):
    """A shared value without deep copy."""

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.value = value

    def __deepcopy__(self, memo):
        if False:
            for i in range(10):
                print('nop')
        return SharedValue(self.value)