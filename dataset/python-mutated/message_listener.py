"""Defines a listener interface for observing certain
state transitions on Message objects.

Also defines a null implementation of this interface.
"""
__author__ = 'robinson@google.com (Will Robinson)'

class MessageListener(object):
    """Listens for modifications made to a message.  Meant to be registered via
  Message._SetListener().

  Attributes:
    dirty:  If True, then calling Modified() would be a no-op.  This can be
            used to avoid these calls entirely in the common case.
  """

    def Modified(self):
        if False:
            i = 10
            return i + 15
        "Called every time the message is modified in such a way that the parent\n    message may need to be updated.  This currently means either:\n    (a) The message was modified for the first time, so the parent message\n        should henceforth mark the message as present.\n    (b) The message's cached byte size became dirty -- i.e. the message was\n        modified for the first time after a previous call to ByteSize().\n        Therefore the parent should also mark its byte size as dirty.\n    Note that (a) implies (b), since new objects start out with a client cached\n    size (zero).  However, we document (a) explicitly because it is important.\n\n    Modified() will *only* be called in response to one of these two events --\n    not every time the sub-message is modified.\n\n    Note that if the listener's |dirty| attribute is true, then calling\n    Modified at the moment would be a no-op, so it can be skipped.  Performance-\n    sensitive callers should check this attribute directly before calling since\n    it will be true most of the time.\n    "
        raise NotImplementedError

class NullMessageListener(object):
    """No-op MessageListener implementation."""

    def Modified(self):
        if False:
            return 10
        pass