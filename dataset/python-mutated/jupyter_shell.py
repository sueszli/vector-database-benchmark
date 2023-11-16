"""An interactive shell for the Jupyter kernel."""
import io
import sys
import builtins
from xonsh.base_shell import BaseShell

class StdJupyterRedirectBuf(io.RawIOBase):
    """Redirects standard I/O buffers to the Jupyter kernel."""

    def __init__(self, redirect):
        if False:
            i = 10
            return i + 15
        self.redirect = redirect
        self.encoding = redirect.encoding
        self.errors = redirect.errors

    def fileno(self):
        if False:
            return 10
        'Returns the file descriptor of the std buffer.'
        return self.redirect.fileno()

    def seek(self, offset, whence=io.SEEK_SET):
        if False:
            i = 10
            return i + 15
        'Sets the location in both the stdbuf and the membuf.'
        raise io.UnsupportedOperation('cannot seek Jupyter redirect')

    def truncate(self, size=None):
        if False:
            i = 10
            return i + 15
        'Truncate both buffers.'
        raise io.UnsupportedOperation('cannot truncate Jupyter redirect')

    def readinto(self, b):
        if False:
            i = 10
            return i + 15
        'Read bytes into buffer from both streams.'
        raise io.UnsupportedOperation('cannot read into Jupyter redirect')

    def write(self, b):
        if False:
            i = 10
            return i + 15
        'Write bytes to kernel.'
        s = b if isinstance(b, str) else b.decode(self.encoding, self.errors)
        self.redirect.write(s)

class StdJupyterRedirect(io.TextIOBase):
    """Redirects a standard I/O stream to the Jupyter kernel."""

    def __init__(self, name, kernel, parent_header=None):
        if False:
            while True:
                i = 10
        "\n        Parameters\n        ----------\n        name : str\n            The name of the buffer in the sys module, e.g. 'stdout'.\n        kernel : XonshKernel\n            Instance of a Jupyter kernel\n        parent_header : dict or None, optional\n            parent header information to pass along with the kernel\n        "
        self._name = name
        self.kernel = kernel
        self.parent_header = parent_header
        self.std = getattr(sys, name)
        self.buffer = StdJupyterRedirectBuf(self)
        setattr(sys, name, self)

    @property
    def encoding(self):
        if False:
            return 10
        'The encoding of the stream'
        env = builtins.__xonsh__.env
        return getattr(self.std, 'encoding', env.get('XONSH_ENCODING'))

    @property
    def errors(self):
        if False:
            for i in range(10):
                print('nop')
        'The encoding errors of the stream'
        env = builtins.__xonsh__.env
        return getattr(self.std, 'errors', env.get('XONSH_ENCODING_ERRORS'))

    @property
    def newlines(self):
        if False:
            print('Hello World!')
        'The newlines of the standard buffer.'
        return self.std.newlines

    def _replace_std(self):
        if False:
            for i in range(10):
                print('nop')
        std = self.std
        if std is None:
            return
        setattr(sys, self._name, std)
        self.std = None

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        self._replace_std()

    def close(self):
        if False:
            return 10
        'Restores the original std stream.'
        self._replace_std()

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.close()

    def write(self, s):
        if False:
            while True:
                i = 10
        'Writes data to the original kernel stream.'
        self.kernel._respond_in_chunks(self._name, s, parent_header=self.parent_header)

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        'Flushes kernel iopub_stream.'
        self.kernel.iopub_stream.flush()

    def fileno(self):
        if False:
            for i in range(10):
                print('nop')
        'Tunnel fileno() calls to the std stream.'
        return self.std.fileno()

    def seek(self, offset, whence=io.SEEK_SET):
        if False:
            print('Hello World!')
        'Seek to a location.'
        raise io.UnsupportedOperation('cannot seek Jupyter redirect')

    def truncate(self, size=None):
        if False:
            return 10
        'Truncate the streams.'
        raise io.UnsupportedOperation('cannot truncate Jupyter redirect')

    def detach(self):
        if False:
            for i in range(10):
                print('nop')
        'This operation is not supported.'
        raise io.UnsupportedOperation('cannot detach a Jupyter redirect')

    def read(self, size=None):
        if False:
            return 10
        'Read from the stream'
        raise io.UnsupportedOperation('cannot read a Jupyter redirect')

    def readline(self, size=-1):
        if False:
            print('Hello World!')
        'Read a line.'
        raise io.UnsupportedOperation('cannot read a line from a Jupyter redirect')

class JupyterShell(BaseShell):
    """A shell for the Jupyter kernel."""

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.kernel = None

    def default(self, line, kernel, parent_header=None):
        if False:
            i = 10
            return i + 15
        'Executes code, but redirects output to Jupyter client'
        stdout = StdJupyterRedirect('stdout', kernel, parent_header)
        stderr = StdJupyterRedirect('stderr', kernel, parent_header)
        with stdout, stderr:
            rtn = super().default(line)
        return rtn