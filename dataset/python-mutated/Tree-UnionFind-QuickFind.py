class UnionFind:

    def __init__(self, n):
        if False:
            while True:
                i = 10
        self.ids = [i for i in range(n)]

    def find(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self.ids[x]

    def union(self, x, y):
        if False:
            i = 10
            return i + 15
        x_id = self.find(x)
        y_id = self.find(y)
        if x_id == y_id:
            return False
        for i in range(len(self.ids)):
            if self.ids[i] == y_id:
                self.ids[i] = x_id
        return True

    def is_connected(self, x, y):
        if False:
            while True:
                i = 10
        return self.find(x) == self.find(y)