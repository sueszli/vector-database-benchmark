from lwe.core.function import Function

class TestFunction(Function):
    __test__ = False

    def __call__(self, word: str, repeats: int, enclose_with: str='') -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Repeat the provided word a number of times.\n\n        :param word: The word to repeat.\n        :type content: str\n        :param repeats: The number of times to repeat the word.\n        :type repeats: int\n        :param enclose_with: Optional string to enclose the final content.\n        :type enclose_with: str, optional\n        :return: A dictionary containing the repeated content.\n        :rtype: dict\n        '
        try:
            repeated_content = ' '.join([word] * repeats)
            enclosed_content = f'{enclose_with}{repeated_content}{enclose_with}'
            output = {'result': enclosed_content, 'message': f'Repeated the word {word} {repeats} times.'}
        except Exception as e:
            output = {'error': str(e)}
        return output