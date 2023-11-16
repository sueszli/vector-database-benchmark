class DictWithoutHasKey(dict):

    def has_key(self, key):
        if False:
            return 10
        raise NotImplementedError('Emulating collections.Mapping which does not have `has_key`.')

def get_dict_without_has_key(**items):
    if False:
        while True:
            i = 10
    return DictWithoutHasKey(**items)