def __instancecheck__(cls, instance):
    if False:
        i = 10
        return i + 15
    return any((cls.__subclasscheck__(c) for c in {subclass, subtype}))