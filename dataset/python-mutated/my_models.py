from datetime import date
from google.appengine.ext import ndb

class LongIntegerProperty(ndb.StringProperty):

    def _validate(self, value):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(value, (int, long)):
            raise TypeError('expected an integer, got %s' % repr(value))

    def _to_base_type(self, value):
        if False:
            print('Hello World!')
        return str(value)

    def _from_base_type(self, value):
        if False:
            for i in range(10):
                print('nop')
        return long(value)

class BoundedLongIntegerProperty(ndb.StringProperty):

    def __init__(self, bits, **kwds):
        if False:
            while True:
                i = 10
        assert isinstance(bits, int)
        assert bits > 0 and bits % 4 == 0
        super(BoundedLongIntegerProperty, self).__init__(**kwds)
        self._bits = bits

    def _validate(self, value):
        if False:
            while True:
                i = 10
        assert -2 ** (self._bits - 1) <= value < 2 ** (self._bits - 1)

    def _to_base_type(self, value):
        if False:
            i = 10
            return i + 15
        if value < 0:
            value += 2 ** self._bits
        assert 0 <= value < 2 ** self._bits
        return '%0*x' % (self._bits // 4, value)

    def _from_base_type(self, value):
        if False:
            return 10
        value = int(value, 16)
        if value >= 2 ** (self._bits - 1):
            value -= 2 ** self._bits
        return value

class MyModel(ndb.Model):
    name = ndb.StringProperty()
    abc = LongIntegerProperty(default=0)
    xyz = LongIntegerProperty(repeated=True)

class FuzzyDate(object):

    def __init__(self, first, last=None):
        if False:
            print('Hello World!')
        assert isinstance(first, date)
        assert last is None or isinstance(last, date)
        self.first = first
        self.last = last or first

class FuzzyDateModel(ndb.Model):
    first = ndb.DateProperty()
    last = ndb.DateProperty()

class FuzzyDateProperty(ndb.StructuredProperty):

    def __init__(self, **kwds):
        if False:
            i = 10
            return i + 15
        super(FuzzyDateProperty, self).__init__(FuzzyDateModel, **kwds)

    def _validate(self, value):
        if False:
            while True:
                i = 10
        assert isinstance(value, FuzzyDate)

    def _to_base_type(self, value):
        if False:
            for i in range(10):
                print('nop')
        return FuzzyDateModel(first=value.first, last=value.last)

    def _from_base_type(self, value):
        if False:
            while True:
                i = 10
        return FuzzyDate(value.first, value.last)

class MaybeFuzzyDateProperty(FuzzyDateProperty):

    def _validate(self, value):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, date):
            return FuzzyDate(value)

class HistoricPerson(ndb.Model):
    name = ndb.StringProperty()
    birth = FuzzyDateProperty()
    death = FuzzyDateProperty()
    event_dates = FuzzyDateProperty(repeated=True)
    event_names = ndb.StringProperty(repeated=True)