class MyEmptyPerson:
    pass
print(MyEmptyPerson)
print(MyEmptyPerson())

class Person:

    def __init__(self, name, surname, alias='Sin alias'):
        if False:
            return 10
        self.full_name = f'{name} {surname} ({alias})'
        self.__name = name

    def get_name(self):
        if False:
            i = 10
            return i + 15
        return self.__name

    def walk(self):
        if False:
            print('Hello World!')
        print(f'{self.full_name} está caminando')
my_person = Person('Brais', 'Moure')
print(my_person.full_name)
print(my_person.get_name())
my_person.walk()
my_other_person = Person('Brais', 'Moure', 'MoureDev')
print(my_other_person.full_name)
my_other_person.walk()
my_other_person.full_name = 'Héctor de León (El loco de los perros)'
print(my_other_person.full_name)
my_other_person.full_name = 666
print(my_other_person.full_name)