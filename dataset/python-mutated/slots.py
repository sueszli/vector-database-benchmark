import inspect

def mro_slots(obj, only_parents: bool=False):
    if False:
        print('Hello World!')
    'Returns a list of all slots of a class and its parents.\n    Args:\n        obj (:obj:`type`): The class or class-instance to get the slots from.\n        only_parents (:obj:`bool`, optional): If ``True``, only the slots of the parents are\n            returned. Defaults to ``False``.\n    '
    cls = obj if inspect.isclass(obj) else obj.__class__
    classes = cls.__mro__[1:] if only_parents else cls.__mro__
    return [attr for cls in classes if hasattr(cls, '__slots__') for attr in cls.__slots__]