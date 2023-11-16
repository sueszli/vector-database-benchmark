import abc
import pydantic

class Class:

    def badAllowed(this):
        if False:
            return 10
        pass

    def stillBad(this):
        if False:
            for i in range(10):
                print('nop')
        pass
    if False:

        def badAllowed(this):
            if False:
                return 10
            pass

        def stillBad(this):
            if False:
                print('Hello World!')
            pass

    @pydantic.validator
    def badAllowed(cls, my_field: str) -> str:
        if False:
            while True:
                i = 10
        pass

    @pydantic.validator
    def stillBad(cls, my_field: str) -> str:
        if False:
            return 10
        pass

    @pydantic.validator('my_field')
    def badAllowed(cls, my_field: str) -> str:
        if False:
            print('Hello World!')
        pass

    @pydantic.validator('my_field')
    def stillBad(cls, my_field: str) -> str:
        if False:
            i = 10
            return i + 15
        pass

    @classmethod
    def badAllowed(cls):
        if False:
            for i in range(10):
                print('nop')
        pass

    @classmethod
    def stillBad(cls):
        if False:
            while True:
                i = 10
        pass

    @abc.abstractclassmethod
    def badAllowed(cls):
        if False:
            return 10
        pass

    @abc.abstractclassmethod
    def stillBad(cls):
        if False:
            i = 10
            return i + 15
        pass

class PosOnlyClass:

    def badAllowed(this, blah, /, self, something: str):
        if False:
            i = 10
            return i + 15
        pass

    def stillBad(this, blah, /, self, something: str):
        if False:
            for i in range(10):
                print('nop')
        pass