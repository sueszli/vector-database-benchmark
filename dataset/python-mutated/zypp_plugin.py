"""
Related to zypp_plugins_test.py module.
"""

class Plugin:
    """
    Bogus module for Zypp Plugins tests.
    """

    def ack(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Acknowledge that the plugin had finished the transaction\n        Returns:\n\n        '

    def main(self):
        if False:
            print('Hello World!')
        '\n        Register plugin\n        Returns:\n\n        '

class BogusIO:
    """
    Read/write logger.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.content = list()
        self.closed = False

    def __str__(self):
        if False:
            return 10
        return '\n'.join(self.content)

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        (self.path, self.mode) = args
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        self.close()

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def write(self, data):
        if False:
            while True:
                i = 10
        '\n        Simulate writing data\n        Args:\n            data:\n\n        Returns:\n\n        '
        self.content.append(data)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Simulate closing the IO object.\n        Returns:\n\n        '
        self.closed = True