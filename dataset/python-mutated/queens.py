"""
N queens problem.

The (well-known) problem is due to Niklaus Wirth.

This solution is inspired by Dijkstra (Structured Programming).  It is
a classic recursive backtracking approach.
"""
N = 8

class Queens:

    def __init__(self, n=N):
        if False:
            return 10
        self.n = n
        self.reset()

    def reset(self):
        if False:
            print('Hello World!')
        n = self.n
        self.y = [None] * n
        self.row = [0] * n
        self.up = [0] * (2 * n - 1)
        self.down = [0] * (2 * n - 1)
        self.nfound = 0

    def solve(self, x=0):
        if False:
            for i in range(10):
                print('nop')
        for y in range(self.n):
            if self.safe(x, y):
                self.place(x, y)
                if x + 1 == self.n:
                    self.display()
                else:
                    self.solve(x + 1)
                self.remove(x, y)

    def safe(self, x, y):
        if False:
            return 10
        return not self.row[y] and (not self.up[x - y]) and (not self.down[x + y])

    def place(self, x, y):
        if False:
            i = 10
            return i + 15
        self.y[x] = y
        self.row[y] = 1
        self.up[x - y] = 1
        self.down[x + y] = 1

    def remove(self, x, y):
        if False:
            print('Hello World!')
        self.y[x] = None
        self.row[y] = 0
        self.up[x - y] = 0
        self.down[x + y] = 0
    silent = 0

    def display(self):
        if False:
            i = 10
            return i + 15
        self.nfound = self.nfound + 1
        if self.silent:
            return
        print('+-' + '--' * self.n + '+')
        for y in range(self.n - 1, -1, -1):
            print('|', end=' ')
            for x in range(self.n):
                if self.y[x] == y:
                    print('Q', end=' ')
                else:
                    print('.', end=' ')
            print('|')
        print('+-' + '--' * self.n + '+')

def main():
    if False:
        print('Hello World!')
    import sys
    silent = 0
    n = N
    if sys.argv[1:2] == ['-n']:
        silent = 1
        del sys.argv[1]
    if sys.argv[1:]:
        n = int(sys.argv[1])
    q = Queens(n)
    q.silent = silent
    q.solve()
    print('Found', q.nfound, 'solutions.')
if __name__ == '__main__':
    main()