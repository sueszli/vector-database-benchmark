from robot.api.deco import keyword

class InvalidKeywords:

    @keyword('Invalid embedded ${args}')
    def invalid_embedded(self):
        if False:
            i = 10
            return i + 15
        pass

    def duplicate_name(self):
        if False:
            i = 10
            return i + 15
        pass

    def duplicateName(self):
        if False:
            return 10
        pass

    @keyword('Same ${embedded}')
    def dupe_with_embedded_1(self, arg):
        if False:
            i = 10
            return i + 15
        pass

    @keyword('same ${match}')
    def dupe_with_embedded_2(self, arg):
        if False:
            while True:
                i = 10
        'This is an error only at run time.'
        pass