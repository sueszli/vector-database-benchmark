class PythonVarArgsConstructor:

    def __init__(self, mandatory, *varargs):
        if False:
            print('Hello World!')
        self.mandatory = mandatory
        self.varargs = varargs

    def get_args(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.mandatory, ' '.join(self.varargs))