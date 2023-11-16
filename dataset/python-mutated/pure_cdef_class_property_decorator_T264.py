class Prop(object):
    """
    >>> p = Prop()
    >>> p.prop
    GETTING 'None'
    >>> p.prop = 1
    SETTING '1' (previously: 'None')
    >>> p.prop
    GETTING '1'
    1
    >>> p.prop = 2
    SETTING '2' (previously: '1')
    >>> p.prop
    GETTING '2'
    2
    >>> del p.prop
    DELETING '2'
    >>> p.prop
    GETTING 'None'
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._value = None

    @property
    def prop(self):
        if False:
            return 10
        print('FAIL')
        return 0

    @prop.getter
    def prop(self):
        if False:
            return 10
        print('FAIL')

    @property
    def prop(self):
        if False:
            print('Hello World!')
        print("GETTING '%s'" % self._value)
        return self._value

    @prop.setter
    def prop(self, value):
        if False:
            return 10
        print("SETTING '%s' (previously: '%s')" % (value, self._value))
        self._value = value

    @prop.deleter
    def prop(self):
        if False:
            print('Hello World!')
        print("DELETING '%s'" % self._value)
        self._value = None