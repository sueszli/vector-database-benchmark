from mrjob.job import MRJob

class SalesRanker(MRJob):

    def within_past_week(self, timestamp):
        if False:
            return 10
        'Return True if timestamp is within past week, False otherwise.'
        ...

    def mapper(self, _, line):
        if False:
            i = 10
            return i + 15
        'Parse each log line, extract and transform relevant lines.\n\n        Emit key value pairs of the form:\n\n        (foo, p1), 2\n        (bar, p1), 2\n        (bar, p1), 1\n        (foo, p2), 3\n        (bar, p3), 10\n        (foo, p4), 1\n        '
        (timestamp, product_id, category, quantity) = line.split('\t')
        if self.within_past_week(timestamp):
            yield ((category, product_id), quantity)

    def reducer(self, key, values):
        if False:
            return 10
        'Sum values for each key.\n\n        (foo, p1), 2\n        (bar, p1), 3\n        (foo, p2), 3\n        (bar, p3), 10\n        (foo, p4), 1\n        '
        yield (key, sum(values))

    def mapper_sort(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        'Construct key to ensure proper sorting.\n\n        Transform key and value to the form:\n\n        (foo, 2), p1\n        (bar, 3), p1\n        (foo, 3), p2\n        (bar, 10), p3\n        (foo, 1), p4\n\n        The shuffle/sort step of MapReduce will then do a\n        distributed sort on the keys, resulting in:\n\n        (category1, 1), product4\n        (category1, 2), product1\n        (category1, 3), product2\n        (category2, 3), product1\n        (category2, 7), product3\n        '
        (category, product_id) = key
        quantity = value
        yield ((category, quantity), product_id)

    def reducer_identity(self, key, value):
        if False:
            i = 10
            return i + 15
        yield (key, value)

    def steps(self):
        if False:
            i = 10
            return i + 15
        'Run the map and reduce steps.'
        return [self.mr(mapper=self.mapper, reducer=self.reducer), self.mr(mapper=self.mapper_sort, reducer=self.reducer_identity)]
if __name__ == '__main__':
    SalesRanker.run()