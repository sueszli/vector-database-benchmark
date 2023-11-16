from collections.abc import MutableMapping

class SettingsOverride(MutableMapping):
    """ This class can be used to override config temporarily
        Basically it returns config[key] if key isn't found in internal dict

        Typical usage:

        settings = SettingsOverride(config.setting)
        settings["option"] = "value"
    """

    def __init__(self, orig_settings, *args, **kwargs):
        if False:
            return 10
        self.orig_settings = orig_settings
        self._dict = dict()
        for (k, v) in dict(*args, **kwargs).items():
            self[k] = v

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        try:
            return self._dict[key]
        except KeyError:
            return self.orig_settings[key]

    def __setitem__(self, key, value):
        if False:
            return 10
        self._dict[key] = value

    def __delitem__(self, key):
        if False:
            return 10
        try:
            del self._dict[key]
        except KeyError:
            pass

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._dict)

    def __iter__(self):
        if False:
            return 10
        return iter(self._dict)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        d = self.orig_settings.copy()
        d.update(self._dict)
        return repr(d)