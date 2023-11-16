def mock_get_ipython():
    if False:
        print('Hello World!')
    "Mock an ipython environment w/o setting up real ipython kernel.\n\n  Each entering of get_ipython() invocation will have the prompt increased by\n  one. Grouping arbitrary python code into separate cells using `with` clause.\n\n  Examples::\n\n    # Usage, before each test function, prepend:\n    @patch('IPython.get_ipython', new_callable=mock_get_ipython)\n\n    # In the test function's signature, add an argument for the patch, e.g.:\n    def some_test(self, cell):\n\n    # Group lines of code into a cell using the argument:\n    with cell:\n      # arbitrary python code\n      # ...\n      # arbitrary python code\n\n    # Next cell with prompt increased by one:\n    with cell:  # Auto-incremental\n      # arbitrary python code\n      # ...\n      # arbitrary python code\n  "

    class MockedGetIpython(object):

        def __init__(self):
            if False:
                print('Hello World!')
            self._execution_count = 0
            self.config = {'IPKernelApp': 'mock'}

        def __call__(self):
            if False:
                while True:
                    i = 10
            return self

        @property
        def execution_count(self):
            if False:
                print('Hello World!')
            'Execution count always starts from 1 and is constant within a cell.'
            return self._execution_count

        def __enter__(self):
            if False:
                print('Hello World!')
            'Marks entering of a cell/prompt.'
            self._execution_count = self._execution_count + 1

        def __exit__(self, exc_type, exc_value, traceback):
            if False:
                return 10
            'Marks exiting of a cell/prompt.'
            pass
    return MockedGetIpython()