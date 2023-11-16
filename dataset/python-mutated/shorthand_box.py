from dynaconf.vendor.box.box import Box

class SBox(Box):
    """
    ShorthandBox (SBox) allows for
    property access of `dict` `json` and `yaml`
    """
    _protected_keys = dir({}) + ['to_dict', 'to_json', 'to_yaml', 'json', 'yaml', 'from_yaml', 'from_json', 'dict', 'toml', 'from_toml', 'to_toml']

    @property
    def dict(self):
        if False:
            print('Hello World!')
        return self.to_dict()

    @property
    def json(self):
        if False:
            for i in range(10):
                print('nop')
        return self.to_json()

    @property
    def yaml(self):
        if False:
            while True:
                i = 10
        return self.to_yaml()

    @property
    def toml(self):
        if False:
            for i in range(10):
                print('nop')
        return self.to_toml()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<ShorthandBox: {0}>'.format(str(self.to_dict()))

    def copy(self):
        if False:
            return 10
        return SBox(super(SBox, self).copy())

    def __copy__(self):
        if False:
            for i in range(10):
                print('nop')
        return SBox(super(SBox, self).copy())