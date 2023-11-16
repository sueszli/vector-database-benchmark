class CustomLen:

    def __init__(self, length):
        if False:
            i = 10
            return i + 15
        self._length = length

    def __len__(self):
        if False:
            return 10
        return self._length

class LengthMethod:

    def length(self):
        if False:
            print('Hello World!')
        return 40

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'length()'

class SizeMethod:

    def size(self):
        if False:
            print('Hello World!')
        return 41

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'size()'

class LengthAttribute:
    length = 42

    def __str__(self):
        if False:
            print('Hello World!')
        return 'length'

def get_variables():
    if False:
        i = 10
        return i + 15
    return dict(CUSTOM_LEN_0=CustomLen(0), CUSTOM_LEN_1=CustomLen(1), CUSTOM_LEN_2=CustomLen(2), CUSTOM_LEN_3=CustomLen(3), LENGTH_METHOD=LengthMethod(), SIZE_METHOD=SizeMethod(), LENGTH_ATTRIBUTE=LengthAttribute())