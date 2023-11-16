import math
import time
import threading
try:
    import queue
except ImportError:
    import Queue as queue
history_file = 'dowstocks.csv'

def minutes(tm):
    if False:
        return 10
    am_pm = tm[-2:]
    fields = tm[:-2].split(':')
    hour = int(fields[0])
    minute = int(fields[1])
    if hour == 12:
        hour = 0
    if am_pm == 'pm':
        hour += 12
    return hour * 60 + minute

def minutes_to_str(m):
    if False:
        i = 10
        return i + 15
    (frac, m) = math.modf(m)
    hours = m // 60
    minutes = m % 60
    seconds = frac * 60
    return '%02d:%02d.%02.f' % (hours, minutes, seconds)

def read_history(filename):
    if False:
        return 10
    result = []
    for line in open(filename):
        str_fields = line.strip().split(',')
        fields = [eval(x) for x in str_fields]
        fields[3] = minutes(fields[3])
        result.append(fields)
    return result

def csv_record(fields):
    if False:
        return 10
    s = '"%s",%0.2f,"%s","%s",%0.2f,%0.2f,%0.2f,%0.2f,%d' % tuple(fields)
    return s

class StockTrack(object):

    def __init__(self, name):
        if False:
            return 10
        self.name = name
        self.history = []
        self.price = 0
        self.time = 0
        self.index = 0
        self.open = 0
        self.low = 0
        self.high = 0
        self.volume = 0
        self.initial = 0
        self.change = 0
        self.date = ''

    def add_data(self, record):
        if False:
            i = 10
            return i + 15
        self.history.append(record)

    def reset(self, time):
        if False:
            for i in range(10):
                print('nop')
        self.time = time
        self.history.sort(key=lambda t: t[3])
        self.index = 0
        while self.index < len(self.history):
            if self.history[self.index][3] > time:
                break
            self.index += 1
        self.open = self.history[0][5]
        self.initial = self.history[0][1] - self.history[0][4]
        self.date = self.history[0][2]
        self.update()
        self.low = self.price
        self.high = self.price

    def interpolate(self, field):
        if False:
            print('Hello World!')
        first = self.history[self.index][field]
        next = self.history[self.index + 1][field]
        first_t = self.history[self.index][3]
        next_t = self.history[self.index + 1][3]
        try:
            slope = (next - first) / (next_t - first_t)
            return first + slope * (self.time - first_t)
        except ZeroDivisionError:
            return first

    def update(self):
        if False:
            return 10
        self.price = round(self.interpolate(1), 2)
        self.volume = int(self.interpolate(-1))
        if self.price < self.low:
            self.low = self.price
        if self.price >= self.high:
            self.high = self.price
        self.change = self.price - self.initial

    def incr(self, dt):
        if False:
            for i in range(10):
                print('nop')
        self.time += dt
        if self.index < len(self.history) - 2:
            while self.index < len(self.history) - 2 and self.time >= self.history[self.index + 1][3]:
                self.index += 1
        self.update()

    def make_record(self):
        if False:
            while True:
                i = 10
        return [self.name, round(self.price, 2), self.date, minutes_to_str(self.time), round(self.change, 2), self.open, round(self.high, 2), round(self.low, 2), self.volume]

class MarketSimulator(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.stocks = {}
        self.prices = {}
        self.time = 0
        self.observers = []

    def register(self, observer):
        if False:
            while True:
                i = 10
        self.observers.append(observer)

    def publish(self, record):
        if False:
            i = 10
            return i + 15
        for obj in self.observers:
            obj.update(record)

    def add_history(self, filename):
        if False:
            for i in range(10):
                print('nop')
        hist = read_history(filename)
        for record in hist:
            if record[0] not in self.stocks:
                self.stocks[record[0]] = StockTrack(record[0])
            self.stocks[record[0]].add_data(record)

    def reset(self, time):
        if False:
            return 10
        self.time = time
        for s in list(self.stocks.values()):
            s.reset(time)

    def run(self, dt):
        if False:
            while True:
                i = 10
        for s in self.stocks:
            self.prices[s] = self.stocks[s].price
            self.publish(self.stocks[s].make_record())
        while self.time < 1000:
            for s in self.stocks:
                self.stocks[s].incr(dt / 60.0)
                if self.stocks[s].price != self.prices[s]:
                    self.prices[s] = self.stocks[s].price
                    self.publish(self.stocks[s].make_record())
            time.sleep(dt)
            self.time += dt / 60.0

class BasicPrinter(object):

    def update(self, record):
        if False:
            i = 10
            return i + 15
        print(csv_record(record))

class LogPrinter(object):

    def __init__(self, filename):
        if False:
            while True:
                i = 10
        self.f = open(filename, 'w')

    def update(self, record):
        if False:
            i = 10
            return i + 15
        self.f.write(csv_record(record) + '\n')
        self.f.flush()
m = MarketSimulator()
m.add_history(history_file)
m.reset(minutes('9:30am'))
m.register(BasicPrinter())
m.register(LogPrinter('stocklog.csv'))
m.run(1)