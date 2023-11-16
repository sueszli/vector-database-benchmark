class Vehicle:

    def general_usage(self):
        if False:
            for i in range(10):
                print('nop')
        print('general use: transporation')

class Car(Vehicle):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        print("I'm car")
        self.wheels = 4
        self.has_roof = True

    def specific_usage(self):
        if False:
            return 10
        self.general_usage()
        print('specific use: commute to work, vacation with family')

class MotorCycle(Vehicle):

    def __init__(self):
        if False:
            while True:
                i = 10
        print("I'm motor cycle")
        self.wheels = 2
        self.has_roof = False

    def specific_usage(self):
        if False:
            while True:
                i = 10
        self.general_usage()
        print('specific use: road trip, racing')
c = Car()
m = MotorCycle()
print(issubclass(Car, MotorCycle))