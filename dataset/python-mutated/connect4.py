__all__ = ['PLAYER1', 'PLAYER2', 'Connect4']
(PLAYER1, PLAYER2) = ('red', 'yellow')

class Connect4:
    """
    A Connect Four game.

    Play moves with :meth:`play`.

    Get past moves with :attr:`moves`.

    Check for a victory with :attr:`winner`.

    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.moves = []
        self.top = [0 for _ in range(7)]
        self.winner = None

    @property
    def last_player(self):
        if False:
            i = 10
            return i + 15
        '\n        Player who played the last move.\n\n        '
        return PLAYER1 if len(self.moves) % 2 else PLAYER2

    @property
    def last_player_won(self):
        if False:
            while True:
                i = 10
        '\n        Whether the last move is winning.\n\n        '
        b = sum((1 << 8 * column + row for (_, column, row) in self.moves[::-2]))
        return any((b & b >> v & b >> 2 * v & b >> 3 * v for v in [1, 7, 8, 9]))

    def play(self, player, column):
        if False:
            for i in range(10):
                print('nop')
        '\n        Play a move in a column.\n\n        Returns the row where the checker lands.\n\n        Raises :exc:`RuntimeError` if the move is illegal.\n\n        '
        if player == self.last_player:
            raise RuntimeError("It isn't your turn.")
        row = self.top[column]
        if row == 6:
            raise RuntimeError('This slot is full.')
        self.moves.append((player, column, row))
        self.top[column] += 1
        if self.winner is None and self.last_player_won:
            self.winner = self.last_player
        return row