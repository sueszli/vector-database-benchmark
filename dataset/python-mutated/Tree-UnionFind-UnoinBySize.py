class UnionFind:

    def __init__(self, n):
        if False:
            while True:
                i = 10
        self.fa = [i for i in range(n)]
        self.size = [1 for i in range(n)]

    def find(self, x):
        if False:
            for i in range(10):
                print('nop')
        while self.fa[x] != x:
            self.fa[x] = self.fa[self.fa[x]]
            x = self.fa[x]
        return x

    def union(self, x, y):
        if False:
            print('Hello World!')
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.size[root_x] < self.size[root_y]:
            self.fa[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        elif self.size[root_x] > self.size[root_y]:
            self.fa[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        else:
            self.fa[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        return True

    def is_connected(self, x, y):
        if False:
            i = 10
            return i + 15
        return self.find(x) == self.find(y)