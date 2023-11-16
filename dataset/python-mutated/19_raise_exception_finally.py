class AdultException(Exception):
    pass

class Person:

    def __init__(self, name, age):
        if False:
            while True:
                i = 10
        self.name = name
        self.age = age

    def get_minor_age(self):
        if False:
            return 10
        if int(self.age) >= 18:
            raise AdultException
        else:
            return self.age

    def display(self):
        if False:
            while True:
                i = 10
        try:
            print(f'age -> {self.get_minor_age()}')
        except AdultException:
            print('Person is an adult')
        finally:
            print(f'name -> {self.name}')
Person('Bhavin', 17).display()
Person('Dhaval', 23).display()