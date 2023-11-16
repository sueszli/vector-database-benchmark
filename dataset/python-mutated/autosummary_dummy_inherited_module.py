from autosummary_dummy_module import Foo

class InheritedAttrClass(Foo):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.subclassattr = 'subclassattr'
        super().__init__()
__all__ = ['InheritedAttrClass']