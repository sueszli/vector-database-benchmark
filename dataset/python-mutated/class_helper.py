from rqalpha.utils.logger import user_system_log
from rqalpha.utils.i18n import gettext as _

def deprecated_property(property_name, instead_property_name):
    if False:
        return 10
    assert property_name != instead_property_name

    def getter(self):
        if False:
            print('Hello World!')
        user_system_log.warn(_('"{}" is deprecated, please use "{}" instead, check the document for more information').format(property_name, instead_property_name))
        return getattr(self, instead_property_name)
    return property(getter)

class CachedProperty:

    def __init__(self, getter):
        if False:
            print('Hello World!')
        self._getter = getter
        self._name = getter.__name__

    def __get__(self, instance, owner):
        if False:
            while True:
                i = 10
        if instance is None:
            return self._getter
        value = self._getter(instance)
        setattr(instance, self._name, value)
        return value
cached_property = CachedProperty