"""Define the abstract class for contextual bandit algorithms."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class BanditAlgorithm(object):
    """A bandit algorithm must be able to do two basic operations.

  1. Choose an action given a context.
  2. Update its internal model given a triple (context, played action, reward).
  """

    def action(self, context):
        if False:
            while True:
                i = 10
        pass

    def update(self, context, action, reward):
        if False:
            for i in range(10):
                print('nop')
        pass