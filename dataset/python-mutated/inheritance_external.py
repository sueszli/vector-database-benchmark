import oso

class User:

    def __init__(self, role, treated):
        if False:
            while True:
                i = 10
        self.role = role
        self._treated = treated

    def treated(self, patient):
        if False:
            i = 10
            return i + 15
        return patient in self._treated

class PatientData:

    def __init__(self, patient):
        if False:
            print('Hello World!')
        self.patient = patient

class Lab(PatientData):
    pass

class Order(PatientData):
    pass

class Test(PatientData):
    pass