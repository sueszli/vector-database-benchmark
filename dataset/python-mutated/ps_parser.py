from .utils import ShellParser

class Parser(ShellParser):
    """Extract text from postscript files using ps2ascii command.
    """

    def extract(self, filename, **kwargs):
        if False:
            i = 10
            return i + 15
        (stdout, _) = self.run(['ps2ascii', filename])
        return stdout