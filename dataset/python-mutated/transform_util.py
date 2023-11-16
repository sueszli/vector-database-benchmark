import re

class TransformUtil:

    @classmethod
    def remove_punctuation(cls, value):
        if False:
            for i in range(10):
                print('nop')
        'Removes !, #, and ?.\n        '
        return re.sub('[!#?]', '', value)

    @classmethod
    def clean_strings(cls, strings, ops):
        if False:
            while True:
                i = 10
        'General purpose method to clean strings.\n\n        Pass in a sequence of strings and the operations to perform.\n        '
        result = []
        for value in strings:
            for function in ops:
                value = function(value)
            result.append(value)
        return result