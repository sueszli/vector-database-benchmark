class Teacher:

    def teachers_action(self):
        if False:
            i = 10
            return i + 15
        print('I can teach')

class Engineer:

    def Engineers_action(self):
        if False:
            return 10
        print('I can code')

class Youtuber:

    def youtubers_action(self):
        if False:
            i = 10
            return i + 15
        print('I can code and teach')

class Person(Teacher, Engineer, Youtuber):
    pass
coder = Person()
coder.teachers_action()
coder.Engineers_action()
coder.youtubers_action()