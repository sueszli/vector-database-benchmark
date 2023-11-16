__all__ = ['VAR']

class Demeter:
    loves = ''

    @property
    def hates(self):
        if False:
            return 10
        return self.loves.upper()

class Variable:
    attr = 'value'
    _attr2 = 'v2'
    attr2 = property(lambda self: self._attr2, lambda self, value: setattr(self, '_attr2', value.upper()))
    demeter = Demeter()

    @property
    def not_settable(self):
        if False:
            for i in range(10):
                print('nop')
        return None
VAR = Variable()