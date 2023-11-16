class UnionFind:

    def __init__(self, n):
        if False:
            i = 10
            return i + 15
        self.fa = [i for i in range(n)]
        self.rank = [1 for i in range(n)]

    def find(self, x):
        if False:
            return 10
        while self.fa[x] != x:
            self.fa[x] = self.fa[self.fa[x]]
            x = self.fa[x]
        return x

    def union(self, x, y):
        if False:
            i = 10
            return i + 15
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            self.fa[root_x] = root_y
        elif self.rank[root_y] > self.rank[root_y]:
            self.fa[root_y] = root_x
        else:
            self.fa[root_x] = root_y
            rank[y] += 1
        return True

    def is_connected(self, x, y):
        if False:
            while True:
                i = 10
        return self.find(x) == self.find(y)