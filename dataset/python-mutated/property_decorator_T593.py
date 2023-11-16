my_property = property

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
    DELETING (previously: '2')

    >>> p.my_prop
    GETTING 'my_prop'
    389

    >>> list(p.generator_prop)
    [42]
    """
    _value = None

    @property
    def prop(self):
        if False:
            i = 10
            return i + 15
        print("GETTING '%s'" % self._value)
        return self._value

    @prop.setter
    def prop(self, value):
        if False:
            while True:
                i = 10
        print("SETTING '%s' (previously: '%s')" % (value, self._value))
        self._value = value

    @prop.deleter
    def prop(self):
        if False:
            i = 10
            return i + 15
        print("DELETING (previously: '%s')" % self._value)
        self._value = None

    @my_property
    def my_prop(self):
        if False:
            for i in range(10):
                print('nop')
        print("GETTING 'my_prop'")
        return 389

    @property
    def generator_prop(self):
        if False:
            i = 10
            return i + 15
        yield 42