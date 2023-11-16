import sys
import types

class OurModule(types.ModuleType):
    attribute_dict = {'valid': 'valid_value', '__package__': __package__, '__name__': __name__}
    if '__loader__' in globals():
        attribute_dict['__loader__'] = __loader__

    def __init__(self, name):
        if False:
            print('Hello World!')
        super(OurModule, self).__init__(name)

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        try:
            return self.attribute_dict[attr]
        except KeyError:
            raise AttributeError(attr)
sys.modules[__name__] = OurModule(__name__)