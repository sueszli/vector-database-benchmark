class Employee:

    def __init__(self, id, name):
        if False:
            return 10
        self.id = id
        self.name = name

    def display(self):
        if False:
            for i in range(10):
                print('nop')
        print(f'ID: {self.id} \nName: {self.name}')
emp = Employee(1, 'coder')
emp.display()
del emp.id
try:
    print(emp.id)
except NameError:
    print('emp.id is not defined')
del emp
try:
    emp.display()
except NameError:
    print('emp is not defined')