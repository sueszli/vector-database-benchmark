from mrjob.job import MRJob

class MRWordCountUtility(MRJob):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(MRWordCountUtility, self).__init__(*args, **kwargs)
        self.chars = 0
        self.words = 0
        self.lines = 0

    def mapper(self, _, line):
        if False:
            print('Hello World!')
        if False:
            yield
        self.chars += len(line) + 1
        self.words += sum((1 for word in line.split() if word.strip()))
        self.lines += 1

    def mapper_final(self):
        if False:
            print('Hello World!')
        yield ('chars', self.chars)
        yield ('words', self.words)
        yield ('lines', self.lines)

    def reducer(self, key, values):
        if False:
            i = 10
            return i + 15
        yield (key, sum(values))
if __name__ == '__main__':
    MRWordCountUtility.run()