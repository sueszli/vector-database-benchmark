"""

D* grid planning

author: Nirnay Roy

See Wikipedia article (https://en.wikipedia.org/wiki/D*)

"""
import math
from sys import maxsize
import matplotlib.pyplot as plt
show_animation = True

class State:

    def __init__(self, x, y):
        if False:
            return 10
        self.x = x
        self.y = y
        self.parent = None
        self.state = '.'
        self.t = 'new'
        self.h = 0
        self.k = 0

    def cost(self, state):
        if False:
            i = 10
            return i + 15
        if self.state == '#' or state.state == '#':
            return maxsize
        return math.sqrt(math.pow(self.x - state.x, 2) + math.pow(self.y - state.y, 2))

    def set_state(self, state):
        if False:
            for i in range(10):
                print('nop')
        '\n        .: new\n        #: obstacle\n        e: oparent of current state\n        *: closed state\n        s: current state\n        '
        if state not in ['s', '.', '#', 'e', '*']:
            return
        self.state = state

class Map:

    def __init__(self, row, col):
        if False:
            print('Hello World!')
        self.row = row
        self.col = col
        self.map = self.init_map()

    def init_map(self):
        if False:
            for i in range(10):
                print('nop')
        map_list = []
        for i in range(self.row):
            tmp = []
            for j in range(self.col):
                tmp.append(State(i, j))
            map_list.append(tmp)
        return map_list

    def get_neighbors(self, state):
        if False:
            print('Hello World!')
        state_list = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if state.x + i < 0 or state.x + i >= self.row:
                    continue
                if state.y + j < 0 or state.y + j >= self.col:
                    continue
                state_list.append(self.map[state.x + i][state.y + j])
        return state_list

    def set_obstacle(self, point_list):
        if False:
            print('Hello World!')
        for (x, y) in point_list:
            if x < 0 or x >= self.row or y < 0 or (y >= self.col):
                continue
            self.map[x][y].set_state('#')

class Dstar:

    def __init__(self, maps):
        if False:
            while True:
                i = 10
        self.map = maps
        self.open_list = set()

    def process_state(self):
        if False:
            for i in range(10):
                print('nop')
        x = self.min_state()
        if x is None:
            return -1
        k_old = self.get_kmin()
        self.remove(x)
        if k_old < x.h:
            for y in self.map.get_neighbors(x):
                if y.h <= k_old and x.h > y.h + x.cost(y):
                    x.parent = y
                    x.h = y.h + x.cost(y)
        elif k_old == x.h:
            for y in self.map.get_neighbors(x):
                if y.t == 'new' or (y.parent == x and y.h != x.h + x.cost(y)) or (y.parent != x and y.h > x.h + x.cost(y)):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
        else:
            for y in self.map.get_neighbors(x):
                if y.t == 'new' or (y.parent == x and y.h != x.h + x.cost(y)):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
                elif y.parent != x and y.h > x.h + x.cost(y):
                    self.insert(y, x.h)
                elif y.parent != x and x.h > y.h + x.cost(y) and (y.t == 'close') and (y.h > k_old):
                    self.insert(y, y.h)
        return self.get_kmin()

    def min_state(self):
        if False:
            while True:
                i = 10
        if not self.open_list:
            return None
        min_state = min(self.open_list, key=lambda x: x.k)
        return min_state

    def get_kmin(self):
        if False:
            while True:
                i = 10
        if not self.open_list:
            return -1
        k_min = min([x.k for x in self.open_list])
        return k_min

    def insert(self, state, h_new):
        if False:
            print('Hello World!')
        if state.t == 'new':
            state.k = h_new
        elif state.t == 'open':
            state.k = min(state.k, h_new)
        elif state.t == 'close':
            state.k = min(state.h, h_new)
        state.h = h_new
        state.t = 'open'
        self.open_list.add(state)

    def remove(self, state):
        if False:
            while True:
                i = 10
        if state.t == 'open':
            state.t = 'close'
        self.open_list.remove(state)

    def modify_cost(self, x):
        if False:
            while True:
                i = 10
        if x.t == 'close':
            self.insert(x, x.parent.h + x.cost(x.parent))

    def run(self, start, end):
        if False:
            return 10
        rx = []
        ry = []
        self.insert(end, 0.0)
        while True:
            self.process_state()
            if start.t == 'close':
                break
        start.set_state('s')
        s = start
        s = s.parent
        s.set_state('e')
        tmp = start
        while tmp != end:
            tmp.set_state('*')
            rx.append(tmp.x)
            ry.append(tmp.y)
            if show_animation:
                plt.plot(rx, ry, '-r')
                plt.pause(0.01)
            if tmp.parent.state == '#':
                self.modify(tmp)
                continue
            tmp = tmp.parent
        tmp.set_state('e')
        return (rx, ry)

    def modify(self, state):
        if False:
            return 10
        self.modify_cost(state)
        while True:
            k_min = self.process_state()
            if k_min >= state.h:
                break

def main():
    if False:
        print('Hello World!')
    m = Map(100, 100)
    (ox, oy) = ([], [])
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10)
    for i in range(-10, 60):
        ox.append(60)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60)
    for i in range(-10, 61):
        ox.append(-10)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40)
        oy.append(60 - i)
    print([(i, j) for (i, j) in zip(ox, oy)])
    m.set_obstacle([(i, j) for (i, j) in zip(ox, oy)])
    start = [10, 10]
    goal = [50, 50]
    if show_animation:
        plt.plot(ox, oy, '.k')
        plt.plot(start[0], start[1], 'og')
        plt.plot(goal[0], goal[1], 'xb')
        plt.axis('equal')
    start = m.map[start[0]][start[1]]
    end = m.map[goal[0]][goal[1]]
    dstar = Dstar(m)
    (rx, ry) = dstar.run(start, end)
    if show_animation:
        plt.plot(rx, ry, '-r')
        plt.show()
if __name__ == '__main__':
    main()