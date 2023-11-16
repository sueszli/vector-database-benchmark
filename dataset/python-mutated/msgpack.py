from warehouse.i18n import LazyString

def object_encode(obj):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(obj, LazyString):
        return str(obj)
    return obj