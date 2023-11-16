"""Structures that allow uniform control over the dispatch process."""
import collections

class DispatchContext(collections.namedtuple('DispatchContext', ('options',))):
    """Allows passing additional parameters to the specific implementations.

  Attributes:
    options: Optional dict of extra arguments that may be required by specific
      implementations.
  """

    def option(self, name):
        if False:
            i = 10
            return i + 15
        return self.options[name]
NO_CTX = DispatchContext(options={})