class Animal:

    def __init__(self, habitat):
        if False:
            while True:
                i = 10
        self.habitat = habitat

    def print_habitat(self):
        if False:
            return 10
        print(self.habitat)

    def sound(self):
        if False:
            for i in range(10):
                print('nop')
        print('Some Animal Sound')

class Dog(Animal):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('Kennel')

    def sound(self):
        if False:
            for i in range(10):
                print('nop')
        print('Woof woof!')
x = Dog()
x.print_habitat()
x.sound()