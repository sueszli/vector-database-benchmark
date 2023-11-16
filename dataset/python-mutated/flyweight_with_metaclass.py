import weakref

class FlyweightMeta(type):

    def __new__(mcs, name, parents, dct):
        if False:
            i = 10
            return i + 15
        '\n        Set up object pool\n\n        :param name: class name\n        :param parents: class parents\n        :param dct: dict: includes class attributes, class methods,\n        static methods, etc\n        :return: new class\n        '
        dct['pool'] = weakref.WeakValueDictionary()
        return super().__new__(mcs, name, parents, dct)

    @staticmethod
    def _serialize_params(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Serialize input parameters to a key.\n        Simple implementation is just to serialize it as a string\n        '
        args_list = list(map(str, args))
        args_list.extend([str(kwargs), cls.__name__])
        key = ''.join(args_list)
        return key

    def __call__(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        key = FlyweightMeta._serialize_params(cls, *args, **kwargs)
        pool = getattr(cls, 'pool', {})
        instance = pool.get(key)
        if instance is None:
            instance = super().__call__(*args, **kwargs)
            pool[key] = instance
        return instance

class Card2(metaclass=FlyweightMeta):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass
if __name__ == '__main__':
    instances_pool = getattr(Card2, 'pool')
    cm1 = Card2('10', 'h', a=1)
    cm2 = Card2('10', 'h', a=1)
    cm3 = Card2('10', 'h', a=2)
    assert cm1 == cm2 and cm1 != cm3
    assert cm1 is cm2 and cm1 is not cm3
    assert len(instances_pool) == 2
    del cm1
    assert len(instances_pool) == 2
    del cm2
    assert len(instances_pool) == 1
    del cm3
    assert len(instances_pool) == 0