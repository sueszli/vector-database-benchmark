import dill

class RestrictedType:

    def __bool__(*args, **kwargs):
        if False:
            print('Hello World!')
        raise Exception('Restricted function')
    __eq__ = __lt__ = __le__ = __ne__ = __gt__ = __ge__ = __hash__ = __bool__
glob_obj = RestrictedType()

def restricted_func():
    if False:
        while True:
            i = 10
    a = glob_obj

def test_function_with_restricted_object():
    if False:
        print('Hello World!')
    deserialized = dill.loads(dill.dumps(restricted_func, recurse=True))
if __name__ == '__main__':
    test_function_with_restricted_object()