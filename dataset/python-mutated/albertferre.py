class Game:

    def __init__(self):
        if False:
            return 10
        self.p1_score = 0
        self.p2_score = 0
        self.finished = False
        self.leading = 'None'
        self.scoring = dict({0: 'Love', 1: '15', 2: '30', 3: '40'})

    def p1_win(self):
        if False:
            while True:
                i = 10
        if not self.finished:
            self.p1_score = self.p1_score + 1
            self.update_leading_player()
        else:
            print('Error: game finished')

    def p2_win(self):
        if False:
            i = 10
            return i + 15
        if not self.finished:
            self.p2_score = self.p2_score + 1
            self.update_leading_player()
        else:
            print('Error: game finished')

    def points_seq(self, p_list, verbose=True):
        if False:
            i = 10
            return i + 15
        for p in p_list:
            if p == 'P1':
                self.p1_win()
            elif p == 'P2':
                self.p2_win()
            else:
                print('Error: invalid value')
            if verbose & ~self.finished:
                print(self)

    def update_leading_player(self):
        if False:
            for i in range(10):
                print('nop')
        if self.p1_score > self.p2_score:
            self.leading = 'P1'
        elif self.p1_score < self.p2_score:
            self.leading = 'P2'
        else:
            self.leading = 'None'

    def reset_game(self):
        if False:
            return 10
        self.p1_score = 0
        self.p2_score = 0
        self.finished = False

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if max(self.p1_score, self.p2_score) > 3:
            if self.p1_score == self.p2_score:
                string = 'Deuce'
            elif abs(self.p1_score - self.p2_score) > 1:
                string = f'Ha ganado el {self.leading}'
                self.finished = True
            else:
                string = f'Ventaja {self.leading}'
        else:
            string = f'{self.scoring[self.p1_score]} - {self.scoring[self.p2_score]}'
        return string
g = Game()
g.points_seq(['P1', 'P1', 'P2', 'P2', 'P1', 'P2', 'P1', 'P1'])
g.reset_game()
g.points_seq(['P1', 'P1', 'P2', 'P2', 'P2', 'P2', 'P1', 'P1'])
g.reset_game()
g.points_seq(['P1', 'P1', 'P2', 'P2', 'P2', 'P1', 'P2', 'P1', 'P1', 'P2', 'P2', 'P2'])