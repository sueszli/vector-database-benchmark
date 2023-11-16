class Duck:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._password = 'password'

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Daffy'