def deprecated():
    if False:
        print('Hello World!')
    '*DEPRECATED*'

def deprecated_with_message():
    if False:
        for i in range(10):
            print('nop')
    '*DEPRECATED for some good reason!* Yes it is. For sure.'

def no_deprecation_whatsoever():
    if False:
        for i in range(10):
            print('nop')
    pass

def silent_deprecation():
    if False:
        return 10
    "*Deprecated* but not yet loudly.\n\n    RF and Libdoc don't consider this being deprecated.\n    "