import csv

def read_rides_as_tuples(filename):
    if False:
        print('Hello World!')
    '\n    Read the bus ride data as a list of tuples\n    '
    records = []
    with open(filename) as f:
        rows = csv.reader(f)
        headings = next(rows)
        for row in rows:
            route = row[0]
            date = row[1]
            daytype = row[2]
            rides = int(row[3])
            record = (route, date, daytype, rides)
            records.append(record)
    return records

def read_rides_as_dicts(filename):
    if False:
        while True:
            i = 10
    '\n    Read the bus ride data as a list of dicts\n    '
    records = []
    with open(filename) as f:
        rows = csv.reader(f)
        headings = next(rows)
        for row in rows:
            route = row[0]
            date = row[1]
            daytype = row[2]
            rides = int(row[3])
            record = {'route': route, 'date': date, 'daytype': daytype, 'rides': rides}
            records.append(record)
    return records

class Row:

    def __init__(self, route, date, daytype, rides):
        if False:
            print('Hello World!')
        self.route = route
        self.date = date
        self.daytype = daytype
        self.rides = rides

def read_rides_as_instances(filename):
    if False:
        while True:
            i = 10
    '\n    Read the bus ride data as a list of instances\n    '
    records = []
    with open(filename) as f:
        rows = csv.reader(f)
        headings = next(rows)
        for row in rows:
            route = row[0]
            date = row[1]
            daytype = row[2]
            rides = int(row[3])
            record = Row(route, date, daytype, rides)
            records.append(record)
    return records
if __name__ == '__main__':
    import tracemalloc
    tracemalloc.start()
    read_rides = read_rides_as_tuples
    rides = read_rides('../../Data/ctabus.csv')
    print('Memory Use: Current %d, Peak %d' % tracemalloc.get_traced_memory())