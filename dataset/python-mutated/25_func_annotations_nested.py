from __future__ import annotations

def foo():
    if False:
        print('Hello World!')
    A = 1

    class C:

        @classmethod
        def f(cls, x: A) -> C:
            if False:
                print('Hello World!')
            y: A = 1
            return cls()