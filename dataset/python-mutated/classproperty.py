class classproperty(object):
    """Class property
    """

    def __init__(self, fget):
        if False:
            i = 10
            return i + 15
        self.fget = fget

    def __get__(self, instance, owner):
        if False:
            return 10
        return self.fget(owner)