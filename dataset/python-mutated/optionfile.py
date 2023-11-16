import configparser

class Parser(configparser.RawConfigParser):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs['allow_no_value'] = True
        configparser.RawConfigParser.__init__(self, **kwargs)

    def __remove_quotes(self, value):
        if False:
            print('Hello World!')
        quotes = ["'", '"']
        for quote in quotes:
            if len(value) >= 2 and value[0] == value[-1] == quote:
                return value[1:-1]
        return value

    def optionxform(self, key):
        if False:
            i = 10
            return i + 15
        return key.lower().replace('_', '-')

    def get(self, section, option):
        if False:
            return 10
        value = configparser.RawConfigParser.get(self, section, option)
        return self.__remove_quotes(value)