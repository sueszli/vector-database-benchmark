import datetime
import sys
from remoteserver import RemoteServer

class ReturnValues:

    def string(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Hyvä tulos!'

    def integer(self):
        if False:
            print('Hello World!')
        return 42

    def float(self):
        if False:
            while True:
                i = 10
        return 3.14

    def boolean(self):
        if False:
            while True:
                i = 10
        return False

    def datetime(self):
        if False:
            for i in range(10):
                print('nop')
        return datetime.datetime(2023, 9, 14, 17, 30, 23)

    def list(self):
        if False:
            while True:
                i = 10
        return [1, 2, 'lolme']

    def dict(self):
        if False:
            while True:
                i = 10
        return {'a': 1, 'b': [2, 3]}
if __name__ == '__main__':
    RemoteServer(ReturnValues(), *sys.argv[1:])