import datetime

class BasicTimezone(datetime.tzinfo):

    def __init__(self, offset, name):
        if False:
            i = 10
            return i + 15
        self.offset = offset
        self.name = name

    def utcoffset(self, dt):
        if False:
            for i in range(10):
                print('nop')
        return self.offset

    def dst(self, dt):
        if False:
            i = 10
            return i + 15
        return datetime.timedelta(0)