"""Test that shadowing a global with a nested function generates a warning."""
import fu

def bar():
    if False:
        print('Hello World!')

    def baz():
        if False:
            while True:
                i = 10

        def fu():
            if False:
                while True:
                    i = 10
            pass