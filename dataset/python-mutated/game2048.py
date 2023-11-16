"""
Function:
    定义2048小游戏
Author:
    Charles
微信公众号:
    Charles的皮卡丘
"""
import copy
import random
import pygame
'2048游戏'

class Game2048(object):

    def __init__(self, matrix_size=(4, 4), max_score_filepath=None, **kwargs):
        if False:
            print('Hello World!')
        self.matrix_size = matrix_size
        self.max_score_filepath = max_score_filepath
        self.initialize()
    '更新游戏状态'

    def update(self):
        if False:
            print('Hello World!')
        game_matrix_before = copy.deepcopy(self.game_matrix)
        self.move()
        if game_matrix_before != self.game_matrix:
            self.randomGenerateNumber()
        if self.score > self.max_score:
            self.max_score = self.score
    '根据指定的方向, 移动所有数字块'

    def move(self):
        if False:
            print('Hello World!')

        def extract(array):
            if False:
                return 10
            array_new = []
            for item in array:
                if item != 'null':
                    array_new.append(item)
            return array_new

        def merge(array):
            if False:
                while True:
                    i = 10
            score = 0
            if len(array) < 2:
                return (array, score)
            for i in range(len(array) - 1):
                if array[i] == 'null':
                    break
                if array[i] == array[i + 1]:
                    array[i] *= 2
                    array.pop(i + 1)
                    array.append('null')
                    score += array[i]
            return (extract(array), score)
        if self.move_direction is None:
            return
        if self.move_direction == 'up':
            for j in range(self.matrix_size[1]):
                col = []
                for i in range(self.matrix_size[0]):
                    col.append(self.game_matrix[i][j])
                col = extract(col)
                col.reverse()
                (col, score) = merge(col)
                self.score += score
                col.reverse()
                col = col + ['null'] * (self.matrix_size[0] - len(col))
                for i in range(self.matrix_size[0]):
                    self.game_matrix[i][j] = col[i]
        elif self.move_direction == 'down':
            for j in range(self.matrix_size[1]):
                col = []
                for i in range(self.matrix_size[0]):
                    col.append(self.game_matrix[i][j])
                col = extract(col)
                (col, score) = merge(col)
                self.score += score
                col = ['null'] * (self.matrix_size[0] - len(col)) + col
                for i in range(self.matrix_size[0]):
                    self.game_matrix[i][j] = col[i]
        elif self.move_direction == 'left':
            for (idx, row) in enumerate(copy.deepcopy(self.game_matrix)):
                row = extract(row)
                row.reverse()
                (row, score) = merge(row)
                self.score += score
                row.reverse()
                row = row + ['null'] * (self.matrix_size[1] - len(row))
                self.game_matrix[idx] = row
        elif self.move_direction == 'right':
            for (idx, row) in enumerate(copy.deepcopy(self.game_matrix)):
                row = extract(row)
                (row, score) = merge(row)
                self.score += score
                row = ['null'] * (self.matrix_size[1] - len(row)) + row
                self.game_matrix[idx] = row
        self.move_direction = None
    '在新的位置随机生成数字'

    def randomGenerateNumber(self):
        if False:
            print('Hello World!')
        empty_pos = []
        for i in range(self.matrix_size[0]):
            for j in range(self.matrix_size[1]):
                if self.game_matrix[i][j] == 'null':
                    empty_pos.append([i, j])
        (i, j) = random.choice(empty_pos)
        self.game_matrix[i][j] = 2 if random.random() > 0.1 else 4
    '初始化'

    def initialize(self):
        if False:
            return 10
        self.game_matrix = [['null' for _ in range(self.matrix_size[1])] for _ in range(self.matrix_size[0])]
        self.score = 0
        self.max_score = self.readMaxScore()
        self.move_direction = None
        self.randomGenerateNumber()
        self.randomGenerateNumber()
    '设置移动方向'

    def setDirection(self, direction):
        if False:
            return 10
        assert direction in ['up', 'down', 'left', 'right']
        self.move_direction = direction
    '保存最高分'

    def saveMaxScore(self):
        if False:
            for i in range(10):
                print('nop')
        f = open(self.max_score_filepath, 'w', encoding='utf-8')
        f.write(str(self.max_score))
        f.close()
    '读取游戏最高分'

    def readMaxScore(self):
        if False:
            print('Hello World!')
        try:
            f = open(self.max_score_filepath, 'r', encoding='utf-8')
            score = int(f.read().strip())
            f.close()
            return score
        except:
            return 0
    '游戏是否结束'

    @property
    def isgameover(self):
        if False:
            i = 10
            return i + 15
        for i in range(self.matrix_size[0]):
            for j in range(self.matrix_size[1]):
                if self.game_matrix[i][j] == 'null':
                    return False
                if i == self.matrix_size[0] - 1 and j == self.matrix_size[1] - 1:
                    continue
                elif i == self.matrix_size[0] - 1:
                    if self.game_matrix[i][j] == self.game_matrix[i][j + 1]:
                        return False
                elif j == self.matrix_size[1] - 1:
                    if self.game_matrix[i][j] == self.game_matrix[i + 1][j]:
                        return False
                elif self.game_matrix[i][j] == self.game_matrix[i + 1][j] or self.game_matrix[i][j] == self.game_matrix[i][j + 1]:
                    return False
        return True