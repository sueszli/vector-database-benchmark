import datetime
from trashcli.put.clock import PutClock

class FixedClock(PutClock):

    def __init__(self, now_value=None):
        if False:
            for i in range(10):
                print('nop')
        self.now_value = now_value

    def set_clock(self, now_value):
        if False:
            for i in range(10):
                print('nop')
        self.now_value = now_value

    def now(self):
        if False:
            print('Hello World!')
        return self.now_value

    @staticmethod
    def fixet_at_jan_1st_2024():
        if False:
            i = 10
            return i + 15
        return FixedClock(now_value=jan_1st_2024())

def jan_1st_2024():
    if False:
        while True:
            i = 10
    return datetime.datetime(2014, 1, 1, 0, 0, 0)