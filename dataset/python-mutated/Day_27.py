def minimum_index(seq):
    if False:
        print('Hello World!')
    if len(seq) == 0:
        raise ValueError('Cannot get the minimum value index from an empty sequence')
    min_idx = 0
    for i in range(1, len(seq)):
        if seq[i] < seq[min_idx]:
            min_idx = i
    return min_idx

class TestDataEmptyArray(object):

    @staticmethod
    def get_array():
        if False:
            for i in range(10):
                print('nop')
        return list()

class TestDataUniqueValues(object):

    @staticmethod
    def get_array():
        if False:
            print('Hello World!')
        return [5, 2, 8, 3, 1, -6, 9]

    @staticmethod
    def get_expected_result():
        if False:
            while True:
                i = 10
        return 5

class TestDataExactlyTwoDifferentMinimums(object):

    @staticmethod
    def get_array():
        if False:
            for i in range(10):
                print('nop')
        return [5, 2, 8, 3, 1, -6, 9, -6, 10]

    @staticmethod
    def get_expected_result():
        if False:
            i = 10
            return i + 15
        return 5

def TestWithEmptyArray():
    if False:
        while True:
            i = 10
    try:
        seq = TestDataEmptyArray.get_array()
        result = minimum_index(seq)
    except ValueError as e:
        pass
    else:
        assert False

def TestWithUniqueValues():
    if False:
        while True:
            i = 10
    seq = TestDataUniqueValues.get_array()
    assert len(seq) >= 2
    assert len(list(set(seq))) == len(seq)
    expected_result = TestDataUniqueValues.get_expected_result()
    result = minimum_index(seq)
    assert result == expected_result

def TestiWithExactyTwoDifferentMinimums():
    if False:
        for i in range(10):
            print('nop')
    seq = TestDataExactlyTwoDifferentMinimums.get_array()
    assert len(seq) >= 2
    tmp = sorted(seq)
    assert tmp[0] == tmp[1] and (len(tmp) == 2 or tmp[1] < tmp[2])
    expected_result = TestDataExactlyTwoDifferentMinimums.get_expected_result()
    result = minimum_index(seq)
    assert result == expected_result
TestWithEmptyArray()
TestWithUniqueValues()
TestiWithExactyTwoDifferentMinimums()
print('OK')