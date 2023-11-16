class Solution(object):

    def exist(self, board, word):
        if False:
            i = 10
            return i + 15
        visited = [[False for j in xrange(len(board[0]))] for i in xrange(len(board))]
        for i in xrange(len(board)):
            for j in xrange(len(board[0])):
                if self.existRecu(board, word, 0, i, j, visited):
                    return True
        return False

    def existRecu(self, board, word, cur, i, j, visited):
        if False:
            i = 10
            return i + 15
        if cur == len(word):
            return True
        if i < 0 or i >= len(board) or j < 0 or (j >= len(board[0])) or visited[i][j] or (board[i][j] != word[cur]):
            return False
        visited[i][j] = True
        result = self.existRecu(board, word, cur + 1, i + 1, j, visited) or self.existRecu(board, word, cur + 1, i - 1, j, visited) or self.existRecu(board, word, cur + 1, i, j + 1, visited) or self.existRecu(board, word, cur + 1, i, j - 1, visited)
        visited[i][j] = False
        return result