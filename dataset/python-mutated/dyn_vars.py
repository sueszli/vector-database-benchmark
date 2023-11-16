from collections import UserDict
from collections.abc import Mapping

def get_variables(type):
    if False:
        print('Hello World!')
    return {'dict': get_dict, 'mydict': MyDict, 'Mapping': get_MyMapping, 'UserDict': get_UserDict, 'MyUserDict': MyUserDict}[type]()

def get_dict():
    if False:
        while True:
            i = 10
    return {'from dict': 'This From Dict', 'from dict2': 2}

class MyDict(dict):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(from_my_dict='This From My Dict', from_my_dict2=2)

def get_MyMapping():
    if False:
        while True:
            i = 10
    data = {'from Mapping': 'This From Mapping', 'from Mapping2': 2}

    class MyMapping(Mapping):

        def __init__(self, data):
            if False:
                while True:
                    i = 10
            self.data = data

        def __getitem__(self, item):
            if False:
                return 10
            return self.data[item]

        def __len__(self):
            if False:
                return 10
            return len(self.data)

        def __iter__(self):
            if False:
                print('Hello World!')
            return iter(self.data)
    return MyMapping(data)

def get_UserDict():
    if False:
        while True:
            i = 10
    return UserDict({'from UserDict': 'This From UserDict', 'from UserDict2': 2})

class MyUserDict(UserDict):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__({'from MyUserDict': 'This From MyUserDict', 'from MyUserDict2': 2})