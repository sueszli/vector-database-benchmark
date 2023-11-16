class UnionFind:

    def __init__(self, n):
        if False:
            return 10
        self.fa = [i for i in range(n)]

    def find(self, x):
        if False:
            i = 10
            return i + 15
        while self.fa[x] != x:
            self.fa[x] = self.fa[self.fa[x]]
            x = self.fa[x]
        return x

    def union(self, x, y):
        if False:
            while True:
                i = 10
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        self.fa[root_x] = root_y
        return True

    def is_connected(self, x, y):
        if False:
            return 10
        return self.find(x) == self.find(y)