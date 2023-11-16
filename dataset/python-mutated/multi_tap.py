from typing import Dict, Optional
from ciphey.iface import Config, Decoder, ParamSpec, T, U, registry

@registry.register
class Multi_tap(Decoder[str]):

    def decode(self, ctext: T) -> Optional[U]:
        if False:
            print('Hello World!')
        result = ''
        for x in ctext.split():
            if x == self.SPACE_DIGIT:
                result += ' '
            elif not Multi_tap.valid_code_part(x):
                return None
            else:
                result += self.decode_num_to_char(x)
        return result

    @staticmethod
    def valid_code_part(code: str) -> bool:
        if False:
            return 10
        if not code.isdigit():
            return False
        if not Multi_tap.is_all_dup(code):
            return False
        if int(code[0]) not in range(2, 10):
            return False
        if len(code) > 4:
            return False
        return True

    @staticmethod
    def decode_num_to_char(number: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        index = Multi_tap.calculate_index(number)
        return Multi_tap.number_index_to_char(index)

    @staticmethod
    def is_all_dup(code):
        if False:
            print('Hello World!')
        return len(set(code)) == 1

    @staticmethod
    def calculate_index(number: str) -> int:
        if False:
            i = 10
            return i + 15
        first_number_as_int = int(number[0])
        number_index = Multi_tap.get_index_from_first_digit(first_number_as_int)
        num_rest_numbers = len(number) - 1
        number_index += num_rest_numbers
        return number_index

    @staticmethod
    def number_index_to_char(index_number: int) -> str:
        if False:
            for i in range(10):
                print('nop')
        start_ascii_value = ord('A')
        return chr(start_ascii_value + index_number)

    @staticmethod
    def get_index_from_first_digit(first_digit: int) -> int:
        if False:
            i = 10
            return i + 15
        number_index = 0
        if first_digit >= 8:
            number_index += 1
        first_digit -= 2
        number_index += first_digit * 3
        return number_index

    @staticmethod
    def priority() -> float:
        if False:
            while True:
                i = 10
        return 0.05

    def __init__(self, config: Config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.SPACE_DIGIT = '0'

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            for i in range(10):
                print('nop')
        return None

    @staticmethod
    def getTarget() -> str:
        if False:
            while True:
                i = 10
        return 'multi_tap'