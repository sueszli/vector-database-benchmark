class BinaryIndexTree:

    def __init__(self, n):
        if False:
            return 10
        self.size = n
        self.tree = [0 for _ in range(n + 1)]

    def lowbit(self, index):
        if False:
            return 10
        return index & -index

    def update(self, index, delta):
        if False:
            print('Hello World!')
        while index <= self.size:
            self.tree[index] += delta
            index += self.lowbit(index)

    def query(self, index):
        if False:
            return 10
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= self.lowbit(index)
        return res