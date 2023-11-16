class Student(object):

    def __init__(self, name):
        if False:
            print('Hello World!')
        self.name = name

    def __call__(self):
        if False:
            while True:
                i = 10
        print('My name is %s.' % self.name)
s = Student('Michael')
s()