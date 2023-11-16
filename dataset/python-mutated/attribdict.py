"""
Copyright (c) 2014-2023 Maltrail developers (https://github.com/stamparm/maltrail/)
See the file 'LICENSE' for copying permission
"""

class AttribDict(dict):

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        return self.get(name)

    def __setattr__(self, name, value):
        if False:
            print('Hello World!')
        self[name] = value