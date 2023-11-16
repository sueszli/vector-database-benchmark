from picard.config import get_config

class PreservedTags:
    opt_name = 'preserved_tags'

    def __init__(self):
        if False:
            print('Hello World!')
        self._tags = self._from_config()

    def _to_config(self):
        if False:
            i = 10
            return i + 15
        config = get_config()
        config.setting[self.opt_name] = sorted(self._tags)

    def _from_config(self):
        if False:
            while True:
                i = 10
        config = get_config()
        tags = config.setting[self.opt_name]
        return set(filter(bool, map(self._normalize_tag, tags)))

    @staticmethod
    def _normalize_tag(tag):
        if False:
            i = 10
            return i + 15
        return tag.strip().lower()

    def add(self, name):
        if False:
            print('Hello World!')
        self._tags.add(self._normalize_tag(name))
        self._to_config()

    def discard(self, name):
        if False:
            for i in range(10):
                print('nop')
        self._tags.discard(self._normalize_tag(name))
        self._to_config()

    def __contains__(self, key):
        if False:
            while True:
                i = 10
        return self._normalize_tag(key) in self._tags