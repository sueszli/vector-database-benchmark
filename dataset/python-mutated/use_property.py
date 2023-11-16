class Student(object):

    @property
    def score(self):
        if False:
            i = 10
            return i + 15
        return self._score

    @score.setter
    def score(self, value):
        if False:
            return 10
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value
s = Student()
s.score = 60
print('s.score =', s.score)
s.score = 9999