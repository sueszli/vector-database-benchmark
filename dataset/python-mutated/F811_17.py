"""Test that shadowing a global name with a nested function generates a warning."""
import fu

def bar():
    if False:
        print('Hello World!')
    import fu

    def baz():
        if False:
            return 10

        def fu():
            if False:
                while True:
                    i = 10
            pass