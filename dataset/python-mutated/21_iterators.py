class RemoteControl:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.channels = ['HBO', 'cnn', 'abc', 'espn']
        self.index = -1

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __next__(self):
        if False:
            i = 10
            return i + 15
        self.index += 1
        if self.index == len(self.channels):
            raise StopIteration
        return self.channels[self.index]
r = RemoteControl()
itr = iter(r)
print(next(itr))
print(next(itr))
print(next(itr))
print(next(itr))
print(next(itr))