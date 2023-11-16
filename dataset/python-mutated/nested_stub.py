import sys

class Outer:

    class InnerStub:
        ...
    outer_attr_after_inner_stub: int

    class Inner:
        inner_attr: int
    outer_attr: int
if sys.version_info > (3, 7):
    if sys.platform == 'win32':
        assignment = 1

        def function_definition(self):
            if False:
                for i in range(10):
                    print('nop')
            ...

    def f1(self) -> str:
        if False:
            i = 10
            return i + 15
        ...
    if sys.platform != 'win32':

        def function_definition(self):
            if False:
                for i in range(10):
                    print('nop')
            ...
        assignment = 1

    def f2(self) -> str:
        if False:
            i = 10
            return i + 15
        ...
import sys

class Outer:

    class InnerStub:
        ...
    outer_attr_after_inner_stub: int

    class Inner:
        inner_attr: int
    outer_attr: int
if sys.version_info > (3, 7):
    if sys.platform == 'win32':
        assignment = 1

        def function_definition(self):
            if False:
                return 10
            ...

    def f1(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        ...
    if sys.platform != 'win32':

        def function_definition(self):
            if False:
                for i in range(10):
                    print('nop')
            ...
        assignment = 1

    def f2(self) -> str:
        if False:
            return 10
        ...