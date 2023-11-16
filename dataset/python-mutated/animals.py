class Animal(object):

    def run(self):
        if False:
            while True:
                i = 10
        print('Animal is running...')

class Dog(Animal):

    def run(self):
        if False:
            while True:
                i = 10
        print('Dog is running...')

class Cat(Animal):

    def run(self):
        if False:
            print('Hello World!')
        print('Cat is running...')

def run_twice(animal):
    if False:
        return 10
    animal.run()
    animal.run()
a = Animal()
d = Dog()
c = Cat()
print('a is Animal?', isinstance(a, Animal))
print('a is Dog?', isinstance(a, Dog))
print('a is Cat?', isinstance(a, Cat))
print('d is Animal?', isinstance(d, Animal))
print('d is Dog?', isinstance(d, Dog))
print('d is Cat?', isinstance(d, Cat))
run_twice(c)