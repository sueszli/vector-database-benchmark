class NamedExplodingObject(object):
    """An object which has no attributes but produces a more informative
    error message when accessed.

    Parameters
    ----------
    name : str
        The name of the object. This will appear in the error messages.

    Notes
    -----
    One common use for this object is so ensure that an attribute always exists
    even if sometimes it should not be used.
    """

    def __init__(self, name, extra_message=None):
        if False:
            return 10
        self._name = name
        self._extra_message = extra_message

    def __getattr__(self, attr):
        if False:
            return 10
        extra_message = self._extra_message
        raise AttributeError('attempted to access attribute %r of ExplodingObject %r%s' % (attr, self._name), ' ' + extra_message if extra_message is not None else '')

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s(%r%s)' % (type(self).__name__, self._name, ', extra_message=...' if self._extra_message is not None else '')