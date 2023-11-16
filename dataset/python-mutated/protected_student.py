class Student(object):

    def __init__(self, name, score):
        if False:
            print('Hello World!')
        self.__name = name
        self.__score = score

    def get_name(self):
        if False:
            while True:
                i = 10
        return self.__name

    def get_score(self):
        if False:
            while True:
                i = 10
        return self.__score

    def set_score(self, score):
        if False:
            while True:
                i = 10
        if 0 <= score <= 100:
            self.__score = score
        else:
            raise ValueError('bad score')

    def get_grade(self):
        if False:
            return 10
        if self.__score >= 90:
            return 'A'
        elif self.__score >= 60:
            return 'B'
        else:
            return 'C'
bart = Student('Bart Simpson', 59)
print('bart.get_name() =', bart.get_name())
bart.set_score(60)
print('bart.get_score() =', bart.get_score())
print('DO NOT use bart._Student__name:', bart._Student__name)