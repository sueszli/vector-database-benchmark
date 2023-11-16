class KwargsLibrary:

    def one_named(self, named=None):
        if False:
            i = 10
            return i + 15
        return named

    def two_named(self, fst=None, snd=None):
        if False:
            for i in range(10):
                print('nop')
        return '%s, %s' % (fst, snd)

    def four_named(self, a=None, b=None, c=None, d=None):
        if False:
            while True:
                i = 10
        return '%s, %s, %s, %s' % (a, b, c, d)

    def mandatory_and_named(self, a, b, c=None):
        if False:
            print('Hello World!')
        return '%s, %s, %s' % (a, b, c)

    def mandatory_named_and_varargs(self, mandatory, d1=None, d2=None, *varargs):
        if False:
            return 10
        return '%s, %s, %s, %s' % (mandatory, d1, d2, '[%s]' % ', '.join(varargs))