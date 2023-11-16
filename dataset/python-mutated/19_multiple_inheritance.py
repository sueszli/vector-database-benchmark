class Father:

    def skills(self):
        if False:
            print('Hello World!')
        print('gardening,programming')

class Mother:

    def skills(self):
        if False:
            while True:
                i = 10
        print('cooking,art')

class Child(Father, Mother):

    def skills(self):
        if False:
            for i in range(10):
                print('nop')
        Father.skills(self)
        Mother.skills(self)
        print('sports')
c = Child()
c.skills()