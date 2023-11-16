class Student(object):

    def __init__(self, name, score):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.score = score

    def print_score(self):
        if False:
            i = 10
            return i + 15
        print('%s: %s' % (self.name, self.score))

    def get_grade(self):
        if False:
            while True:
                i = 10
        if self.score >= 90:
            return 'A'
        elif self.score >= 60:
            return 'B'
        else:
            return 'C'
bart = Student('Bart Simpson', 59)
lisa = Student('Lisa Simpson', 87)
print('bart.name =', bart.name)
print('bart.score =', bart.score)
bart.print_score()
print('grade of Bart:', bart.get_grade())
print('grade of Lisa:', lisa.get_grade())