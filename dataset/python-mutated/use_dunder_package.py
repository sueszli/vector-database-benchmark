if '__torch_package__' in dir():

    def is_from_package():
        if False:
            while True:
                i = 10
        return True
else:

    def is_from_package():
        if False:
            while True:
                i = 10
        return False