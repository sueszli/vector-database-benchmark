"""Exception for errors raised while interpreting nodes."""

class NodeException(Exception):
    """Base class for errors raised while interpreting nodes."""

    def __init__(self, *msg):
        if False:
            print('Hello World!')
        'Set the error message.'
        super().__init__(*msg)
        self.msg = ' '.join(msg)

    def __str__(self):
        if False:
            return 10
        'Return the message.'
        return repr(self.msg)