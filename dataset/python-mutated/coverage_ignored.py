class Documented:
    """Documented"""

    def ignored1(self):
        if False:
            print('Hello World!')
        pass

    def ignored2(self):
        if False:
            while True:
                i = 10
        pass

    def not_ignored1(self):
        if False:
            return 10
        pass

    def not_ignored2(self):
        if False:
            print('Hello World!')
        pass

class Ignored:
    pass

class NotIgnored:
    pass