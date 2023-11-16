try:
    from UserDict import UserDict
except ImportError:
    from collections import UserDict

def get_user_dict(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    return UserDict(**kwargs)