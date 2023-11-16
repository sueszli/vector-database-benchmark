class CircularDependencyError(RuntimeError):
    """
    An error identifying a circular dependency.
    """

    def __init__(self, names=None):
        if False:
            while True:
                i = 10
        '\n        Creates the CircularDependencyError with a helpful message about the\n        dependency.\n\n        :param names:\n            The names of the nodes that form a dependency circle.\n        '
        if names:
            joined_names = ' -> '.join(names)
            msg = f'Circular dependency detected: {joined_names}'
        else:
            msg = 'Circular dependency detected.'
        super().__init__(msg)