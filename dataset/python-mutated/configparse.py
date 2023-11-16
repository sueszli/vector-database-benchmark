import re

class ConfigValue(object):
    _bool_map = dict(true=True, false=False)

    @staticmethod
    def str(v, key=None):
        if False:
            i = 10
            return i + 15
        return str(v)

    @staticmethod
    def int(v, key=None):
        if False:
            print('Hello World!')
        return int(v)

    @staticmethod
    def float(v, key=None):
        if False:
            return 10
        return float(v)

    @staticmethod
    def bool(v, key=None):
        if False:
            while True:
                i = 10
        if v in (True, False, None):
            return bool(v)
        try:
            return ConfigValue._bool_map[v.lower()]
        except KeyError:
            raise ValueError('Unknown value for %r: %r' % (key, v))

    @staticmethod
    def tuple(v, key=None):
        if False:
            for i in range(10):
                print('nop')
        return tuple(ConfigValue.to_iter(v))

    @staticmethod
    def set(v, key=None):
        if False:
            print('Hello World!')
        return set(ConfigValue.to_iter(v))

    @staticmethod
    def set_of(value_type, delim=','):
        if False:
            return 10

        def parse(v, key=None):
            if False:
                while True:
                    i = 10
            return set((value_type(x) for x in ConfigValue.to_iter(v, delim=delim)))
        return parse

    @staticmethod
    def tuple_of(value_type, delim=','):
        if False:
            for i in range(10):
                print('nop')

        def parse(v, key=None):
            if False:
                return 10
            return tuple((value_type(x) for x in ConfigValue.to_iter(v, delim=delim)))
        return parse

    @staticmethod
    def dict(key_type, value_type, delim=',', kvdelim=':'):
        if False:
            return 10

        def parse(v, key=None):
            if False:
                return 10
            values = (i.partition(kvdelim) for i in ConfigValue.to_iter(v, delim=delim))
            return {key_type(x): value_type(y) for (x, _, y) in values}
        return parse

    @staticmethod
    def choice(**choices):
        if False:
            i = 10
            return i + 15

        def parse_choice(v, key=None):
            if False:
                while True:
                    i = 10
            try:
                return choices[v]
            except KeyError:
                raise ValueError('Unknown option for %r: %r not in %r' % (key, v, choices.keys()))
        return parse_choice

    @staticmethod
    def to_iter(v, delim=','):
        if False:
            while True:
                i = 10
        return (x.strip() for x in v.split(delim) if x)

    @staticmethod
    def timeinterval(v, key=None):
        if False:
            return 10
        from r2.lib.utils import timeinterval_fromstr
        return timeinterval_fromstr(v)
    messages_re = re.compile('"([^"]+)"')

    @staticmethod
    def messages(v, key=None):
        if False:
            return 10
        return ConfigValue.messages_re.findall(v.decode('string_escape'))

    @staticmethod
    def baseplate(baseplate_parser):
        if False:
            while True:
                i = 10

        def adapter(v, key=None):
            if False:
                while True:
                    i = 10
            return baseplate_parser(v)
        return adapter

class ConfigValueParser(dict):

    def __init__(self, raw_data):
        if False:
            return 10
        dict.__init__(self, raw_data)
        self.config_keys = {}
        self.raw_data = raw_data

    def add_spec(self, spec):
        if False:
            print('Hello World!')
        new_keys = []
        for (parser, keys) in spec.iteritems():
            for key in keys:
                assert key not in self.config_keys
                self.config_keys[key] = parser
                new_keys.append(key)
        self._update_values(new_keys)

    def _update_values(self, keys):
        if False:
            return 10
        for key in keys:
            if key not in self.raw_data:
                continue
            value = self.raw_data[key]
            if key in self.config_keys:
                parser = self.config_keys[key]
                value = parser(value, key)
            self[key] = value