from lwe.core.function import Function

class ReverseContent(Function):

    def __call__(self, content: str) -> dict:
        if False:
            return 10
        '\n        Reverse the provided content\n\n        :param content: The content to reverse.\n        :type content: str\n        :return: A dictionary containing the reversed content.\n        :rtype: dict\n        '
        try:
            reversed_content = content[::-1]
            output = {'result': reversed_content, 'message': 'Reversed the content string'}
        except Exception as e:
            output = {'error': str(e)}
        return output