"""
Custom Spyder Outstream class.
"""
from ipykernel.iostream import OutStream

class TTYOutStream(OutStream):
    """Subclass of OutStream that represents a TTY."""

    def __init__(self, session, pub_thread, name, pipe=None, echo=None, *, watchfd=True):
        if False:
            print('Hello World!')
        super().__init__(session, pub_thread, name, pipe, echo=echo, watchfd=watchfd, isatty=True)