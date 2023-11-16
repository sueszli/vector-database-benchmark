from abc import ABC, abstractmethod

class _Methods(ABC):
    """Abstract Base Class for all methods."""

    @abstractmethod
    def q(self):
        if False:
            return 10
        pass

    @abstractmethod
    def u(self):
        if False:
            return 10
        pass

    @abstractmethod
    def bodies(self):
        if False:
            return 10
        pass

    @abstractmethod
    def loads(self):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def mass_matrix(self):
        if False:
            return 10
        pass

    @abstractmethod
    def forcing(self):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def mass_matrix_full(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def forcing_full(self):
        if False:
            i = 10
            return i + 15
        pass

    def _form_eoms(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Subclasses must implement this.')