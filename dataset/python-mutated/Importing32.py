def starImportFailure():
    if False:
        for i in range(10):
            print('nop')
    from doctest import *
    try:
        sys
        print('but it does not')
    except NameError:
        print('and it does')
print('Star import needs to respect __all__', starImportFailure())