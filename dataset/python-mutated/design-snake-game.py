from collections import deque

class SnakeGame(object):

    def __init__(self, width, height, food):
        if False:
            return 10
        '\n        Initialize your data structure here.\n        @param width - screen width\n        @param height - screen height\n        @param food - A list of food positions\n        E.g food = [[1,1], [1,0]] means the first food is positioned at [1,1], the second is at [1,0].\n        :type width: int\n        :type height: int\n        :type food: List[List[int]]\n        '
        self.__width = width
        self.__height = height
        self.__score = 0
        self.__f = 0
        self.__food = food
        self.__snake = deque([(0, 0)])
        self.__direction = {'U': (-1, 0), 'L': (0, -1), 'R': (0, 1), 'D': (1, 0)}
        self.__lookup = {(0, 0)}

    def move(self, direction):
        if False:
            return 10
        "\n        Moves the snake.\n        @param direction - 'U' = Up, 'L' = Left, 'R' = Right, 'D' = Down\n        @return The game's score after the move. Return -1 if game over.\n        Game over when snake crosses the screen boundary or bites its body.\n        :type direction: str\n        :rtype: int\n        "

        def valid(x, y):
            if False:
                print('Hello World!')
            return 0 <= x < self.__height and 0 <= y < self.__width and ((x, y) not in self.__lookup)
        d = self.__direction[direction]
        (x, y) = (self.__snake[-1][0] + d[0], self.__snake[-1][1] + d[1])
        self.__lookup.remove(self.__snake[0])
        tail = self.__snake.popleft()
        if not valid(x, y):
            return -1
        elif self.__f != len(self.__food) and (self.__food[self.__f][0], self.__food[self.__f][1]) == (x, y):
            self.__score += 1
            self.__f += 1
            self.__snake.appendleft(tail)
            self.__lookup.add(tail)
        self.__snake.append((x, y))
        self.__lookup.add((x, y))
        return self.__score