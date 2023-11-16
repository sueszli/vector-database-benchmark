def get_evens(nums: list):
    if False:
        return 10
    return [num for num in nums if num % 2 == 0]

def test_get_evens():
    if False:
        i = 10
        return i + 15
    assert get_evens([1, 3, 4, 6, 8, 9]) == [4, 6, 8]
'On your terminal\nls entr_examples.py | entr python entr_examples.py \n'