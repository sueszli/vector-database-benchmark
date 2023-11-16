class Person:

    def __init__(self, initialAge):
        if False:
            return 10
        self.age = 0
        if initialAge < 0:
            print('Age is not valid, setting age to 0.')
        else:
            self.age = initialAge

    def amIOld(self):
        if False:
            i = 10
            return i + 15
        if age < 13:
            print('You are young.')
        elif 13 <= age < 18:
            print('You are a teenager.')
        elif age >= 18:
            print('You are old.')

    def yearPasses(self):
        if False:
            print('Hello World!')
        global age
        age += 1
t = int(input())