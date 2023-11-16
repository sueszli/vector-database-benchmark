from mrjob.job import MRJob

class RemoveDuplicateUrls(MRJob):

    def mapper(self, _, line):
        if False:
            i = 10
            return i + 15
        yield (line, 1)

    def reducer(self, key, values):
        if False:
            while True:
                i = 10
        total = sum(values)
        if total == 1:
            yield (key, total)

    def steps(self):
        if False:
            print('Hello World!')
        'Run the map and reduce steps.'
        return [self.mr(mapper=self.mapper, reducer=self.reducer)]
if __name__ == '__main__':
    RemoveDuplicateUrls.run()