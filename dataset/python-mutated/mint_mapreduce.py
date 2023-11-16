from mrjob.job import MRJob

class SpendingByCategory(MRJob):

    def __init__(self, categorizer):
        if False:
            return 10
        self.categorizer = categorizer
        ...

    def current_year_month(self):
        if False:
            print('Hello World!')
        'Return the current year and month.'
        ...

    def extract_year_month(self, timestamp):
        if False:
            for i in range(10):
                print('nop')
        'Return the year and month portions of the timestamp.'
        ...

    def handle_budget_notifications(self, key, total):
        if False:
            for i in range(10):
                print('nop')
        'Call notification API if nearing or exceeded budget.'
        ...

    def mapper(self, _, line):
        if False:
            for i in range(10):
                print('nop')
        'Parse each log line, extract and transform relevant lines.\n\n        Emit key value pairs of the form:\n\n        (2016-01, shopping), 25\n        (2016-01, shopping), 100\n        (2016-01, gas), 50\n        '
        (timestamp, category, amount) = line.split('\t')
        period = self.extract_year_month(timestamp)
        if period == self.current_year_month():
            yield ((period, category), amount)

    def reducer(self, key, values):
        if False:
            i = 10
            return i + 15
        'Sum values for each key.\n\n        (2016-01, shopping), 125\n        (2016-01, gas), 50\n        '
        total = sum(values)
        self.handle_budget_notifications(key, total)
        yield (key, sum(values))

    def steps(self):
        if False:
            while True:
                i = 10
        'Run the map and reduce steps.'
        return [self.mr(mapper=self.mapper, reducer=self.reducer)]
if __name__ == '__main__':
    SpendingByCategory.run()