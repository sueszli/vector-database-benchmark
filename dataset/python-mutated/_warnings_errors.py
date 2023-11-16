class DegenerateDataWarning(RuntimeWarning):
    """Warns when data is degenerate and results may not be reliable."""

    def __init__(self, msg=None):
        if False:
            i = 10
            return i + 15
        if msg is None:
            msg = 'Degenerate data encountered; results may not be reliable.'
        self.args = (msg,)

class ConstantInputWarning(DegenerateDataWarning):
    """Warns when all values in data are exactly equal."""

    def __init__(self, msg=None):
        if False:
            i = 10
            return i + 15
        if msg is None:
            msg = 'All values in data are exactly equal; results may not be reliable.'
        self.args = (msg,)

class NearConstantInputWarning(DegenerateDataWarning):
    """Warns when all values in data are nearly equal."""

    def __init__(self, msg=None):
        if False:
            print('Hello World!')
        if msg is None:
            msg = 'All values in data are nearly equal; results may not be reliable.'
        self.args = (msg,)

class FitError(RuntimeError):
    """Represents an error condition when fitting a distribution to data."""

    def __init__(self, msg=None):
        if False:
            print('Hello World!')
        if msg is None:
            msg = 'An error occurred when fitting a distribution to data.'
        self.args = (msg,)