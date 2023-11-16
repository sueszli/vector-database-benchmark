class Student(object):

    def __init__(self, name):
        if False:
            while True:
                i = 10
        self.name = name

    def __str__(self):
        if False:
            return 10
        return 'Student object (name: %s)' % self.name
    __repr__ = __str__
print(Student('Michael'))