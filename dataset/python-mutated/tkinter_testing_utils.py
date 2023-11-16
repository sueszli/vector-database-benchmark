"""Utilities for testing with Tkinter"""
import functools

def run_in_tk_mainloop(delay=1):
    if False:
        for i in range(10):
            print('nop')
    'Decorator for running a test method with a real Tk mainloop.\n\n    This starts a Tk mainloop before running the test, and stops it\n    at the end. This is faster and more robust than the common\n    alternative method of calling .update() and/or .update_idletasks().\n\n    Test methods using this must be written as generator functions,\n    using "yield" to allow the mainloop to process events and "after"\n    callbacks, and then continue the test from that point.\n\n    The delay argument is passed into root.after(...) calls as the number\n    of ms to wait before passing execution back to the generator function.\n\n    This also assumes that the test class has a .root attribute,\n    which is a tkinter.Tk object.\n\n    For example (from test_sidebar.py):\n\n    @run_test_with_tk_mainloop()\n    def test_single_empty_input(self):\n        self.do_input(\'\n\')\n        yield\n        self.assert_sidebar_lines_end_with([\'>>>\', \'>>>\'])\n    '

    def decorator(test_method):
        if False:
            print('Hello World!')

        @functools.wraps(test_method)
        def new_test_method(self):
            if False:
                i = 10
                return i + 15
            test_generator = test_method(self)
            root = self.root
            exception = None

            def after_callback():
                if False:
                    while True:
                        i = 10
                nonlocal exception
                try:
                    next(test_generator)
                except StopIteration:
                    root.quit()
                except Exception as exc:
                    exception = exc
                    root.quit()
                else:
                    root.after(delay, root.after_idle, after_callback)
            root.after(0, root.after_idle, after_callback)
            root.mainloop()
            if exception:
                raise exception
        return new_test_method
    return decorator