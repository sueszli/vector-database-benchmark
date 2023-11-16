class TicTacToe(object):

    def __init__(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize your data structure here.\n        :type n: int\n        '
        self.__size = n
        self.__rows = [[0, 0] for _ in xrange(n)]
        self.__cols = [[0, 0] for _ in xrange(n)]
        self.__diagonal = [0, 0]
        self.__anti_diagonal = [0, 0]

    def move(self, row, col, player):
        if False:
            while True:
                i = 10
        '\n        Player {player} makes a move at ({row}, {col}).\n        @param row The row of the board.\n        @param col The column of the board.\n        @param player The player, can be either 1 or 2.\n        @return The current winning condition, can be either:\n                0: No one wins.\n                1: Player 1 wins.\n                2: Player 2 wins.\n        :type row: int\n        :type col: int\n        :type player: int\n        :rtype: int\n        '
        i = player - 1
        self.__rows[row][i] += 1
        self.__cols[col][i] += 1
        if row == col:
            self.__diagonal[i] += 1
        if col == len(self.__rows) - row - 1:
            self.__anti_diagonal[i] += 1
        if any(self.__rows[row][i] == self.__size, self.__cols[col][i] == self.__size, self.__diagonal[i] == self.__size, self.__anti_diagonal[i] == self.__size):
            return player
        return 0