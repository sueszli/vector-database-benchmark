from sympy.tensor.array.ndim_array import NDimArray

class MutableNDimArray(NDimArray):

    def as_immutable(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('abstract method')

    def as_mutable(self):
        if False:
            print('Hello World!')
        return self

    def _sympy_(self):
        if False:
            return 10
        return self.as_immutable()