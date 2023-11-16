"""
Option validation with exception handling.
"""

class Validator:
    """
    When a given validation function returns True, given exception is raised.
    """

    def __init__(self, validation_function, exception):
        if False:
            return 10
        self.validation_function = validation_function
        self.exception = exception

    def validate(self, *args, **kwargs):
        if False:
            return 10
        if self.validation_function(*args, **kwargs):
            raise self.exception