import json
import re
from superagi.lib.logger import logger
import json5

class JsonCleaner:

    @classmethod
    def clean_boolean(cls, input_str: str=''):
        if False:
            for i in range(10):
                print('nop')
        '\n        Clean the boolean values in the given string.\n\n        Args:\n            input_str (str): The string from which the json section is to be extracted.\n\n        Returns:\n            str: The extracted json section.\n        '
        input_str = re.sub(':\\s*false', ': False', input_str)
        input_str = re.sub(':\\s*true', ': True', input_str)
        return input_str

    @classmethod
    def extract_json_section(cls, input_str: str=''):
        if False:
            return 10
        '\n        Extract the json section from the given string.\n\n        Args:\n            input_str (str): The string from which the json section is to be extracted.\n\n        Returns:\n            str: The extracted json section.\n        '
        try:
            first_brace_index = input_str.index('{')
            final_json = input_str[first_brace_index:]
            last_brace_index = final_json.rindex('}')
            final_json = final_json[:last_brace_index + 1]
            return final_json
        except ValueError:
            pass
        return input_str

    @classmethod
    def extract_json_array_section(cls, input_str: str=''):
        if False:
            print('Hello World!')
        '\n        Extract the json section from the given string.\n\n        Args:\n            input_str (str): The string from which the json section is to be extracted.\n\n        Returns:\n            str: The extracted json section.\n        '
        try:
            first_brace_index = input_str.index('[')
            final_json = input_str[first_brace_index:]
            last_brace_index = final_json.rindex(']')
            final_json = final_json[:last_brace_index + 1]
            return final_json
        except ValueError:
            pass
        return input_str

    @classmethod
    def remove_escape_sequences(cls, string):
        if False:
            print('Hello World!')
        '\n        Remove escape sequences from the given string.\n\n        Args:\n            string (str): The string from which the escape sequences are to be removed.\n\n        Returns:\n            str: The string with escape sequences removed.\n        '
        return string.encode('utf-8').decode('unicode_escape').encode('raw_unicode_escape').decode('utf-8')

    @classmethod
    def balance_braces(cls, json_string: str) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Balance the braces in the given json string.\n\n        Args:\n            json_string (str): The json string to be processed.\n\n        Returns:\n            str: The json string with balanced braces.\n        '
        open_braces_count = json_string.count('{')
        closed_braces_count = json_string.count('}')
        while closed_braces_count > open_braces_count:
            json_string = json_string.rstrip('}')
            closed_braces_count -= 1
        open_braces_count = json_string.count('{')
        closed_braces_count = json_string.count('}')
        if open_braces_count > closed_braces_count:
            json_string += '}' * (open_braces_count - closed_braces_count)
        return json_string