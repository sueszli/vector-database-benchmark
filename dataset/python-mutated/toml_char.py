import string

class TOMLChar(str):

    def __init__(self, c):
        if False:
            i = 10
            return i + 15
        super().__init__()
        if len(self) > 1:
            raise ValueError('A TOML character must be of length 1')
    BARE = string.ascii_letters + string.digits + '-_'
    KV = '= \t'
    NUMBER = string.digits + '+-_.e'
    SPACES = ' \t'
    NL = '\n\r'
    WS = SPACES + NL

    def is_bare_key_char(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Whether the character is a valid bare key name or not.\n        '
        return self in self.BARE

    def is_kv_sep(self) -> bool:
        if False:
            return 10
        '\n        Whether the character is a valid key/value separator or not.\n        '
        return self in self.KV

    def is_int_float_char(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Whether the character if a valid integer or float value character or not.\n        '
        return self in self.NUMBER

    def is_ws(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Whether the character is a whitespace character or not.\n        '
        return self in self.WS

    def is_nl(self) -> bool:
        if False:
            return 10
        '\n        Whether the character is a new line character or not.\n        '
        return self in self.NL

    def is_spaces(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Whether the character is a space or not\n        '
        return self in self.SPACES