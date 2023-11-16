from random import choices, randint

class StringGenerator:
    _seed_letters: str
    _min_length: int
    _max_length: int

    def __init__(self, seed_letters: str, min_length: int, max_length: int):
        if False:
            print('Hello World!')
        self._seed_letters = seed_letters
        self._min_length = min_length
        self._max_length = max_length

    def generate(self) -> str:
        if False:
            print('Hello World!')
        rv_string_length = randint(self._min_length, self._max_length)
        randomized_letters = choices(self._seed_letters, k=rv_string_length)
        return ''.join(randomized_letters)