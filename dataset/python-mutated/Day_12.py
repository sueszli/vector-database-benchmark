class Person:

    def __init__(self, firstName, lastName, idNumber):
        if False:
            while True:
                i = 10
        self.firstName = firstName
        self.lastName = lastName
        self.idNumber = idNumber

    def printPerson(self):
        if False:
            return 10
        print('Name:', self.lastName + ',', self.firstName)
        print('ID:', self.idNumber)

class Student(Person):

    def __init__(self, firstName, lastName, idNumber, scores):
        if False:
            while True:
                i = 10
        Person.__init__(self, firstName, lastName, idNumber)
        self.scores = scores

    def calculate(self):
        if False:
            while True:
                i = 10
        s = 0
        for score in scores:
            s += score
        average = s / len(scores)
        if average < 40:
            return 'T'
        elif average < 55:
            return 'D'
        elif average < 70:
            return 'P'
        elif average < 80:
            return 'A'
        elif average < 90:
            return 'E'
        else:
            return 'O'
line = input().split()
firstName = line[0]
lastName = line[1]
idNum = line[2]
numScores = int(input())
scores = list(map(int, input().split()))
s = Student(firstName, lastName, idNum, scores)
s.printPerson()
print('Grade:', s.calculate())