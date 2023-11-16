from types import MethodType

def is_hypothesis_test(test):
    if False:
        return 10
    if isinstance(test, MethodType):
        return is_hypothesis_test(test.__func__)
    return getattr(test, 'is_hypothesis_test', False)