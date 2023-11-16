"""Python context management helper."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class IdentityContextManager(object):
    """Returns an identity context manager that does nothing.

  This is helpful in setting up conditional `with` statement as below:

  with slim.arg_scope(x) if use_slim_scope else IdentityContextManager():
    do_stuff()

  """

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    def __exit__(self, exec_type, exec_value, traceback):
        if False:
            return 10
        del exec_type
        del exec_value
        del traceback
        return False