"""
*What is this pattern about?
The Borg pattern (also known as the Monostate pattern) is a way to
implement singleton behavior, but instead of having only one instance
of a class, there are multiple instances that share the same state. In
other words, the focus is on sharing state instead of sharing instance
identity.

*What does this example do?
To understand the implementation of this pattern in Python, it is
important to know that, in Python, instance attributes are stored in a
attribute dictionary called __dict__. Usually, each instance will have
its own dictionary, but the Borg pattern modifies this so that all
instances have the same dictionary.
In this example, the __shared_state attribute will be the dictionary
shared between all instances, and this is ensured by assigning
__shared_state to the __dict__ variable when initializing a new
instance (i.e., in the __init__ method). Other attributes are usually
added to the instance's attribute dictionary, but, since the attribute
dictionary itself is shared (which is __shared_state), all other
attributes will also be shared.

*Where is the pattern used practically?
Sharing state is useful in applications like managing database connections:
https://github.com/onetwopunch/pythonDbTemplate/blob/master/database.py

*References:
- https://fkromer.github.io/python-pattern-references/design/#singleton
- https://learning.oreilly.com/library/view/python-cookbook/0596001673/ch05s23.html
- http://www.aleax.it/5ep.html

*TL;DR
Provides singleton-like behavior sharing state between instances.
"""
from typing import Dict

class Borg:
    _shared_state: Dict[str, str] = {}

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.__dict__ = self._shared_state

class YourBorg(Borg):

    def __init__(self, state: str=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        if state:
            self.state = state
        elif not hasattr(self, 'state'):
            self.state = 'Init'

    def __str__(self) -> str:
        if False:
            return 10
        return self.state

def main():
    if False:
        print('Hello World!')
    "\n    >>> rm1 = YourBorg()\n    >>> rm2 = YourBorg()\n\n    >>> rm1.state = 'Idle'\n    >>> rm2.state = 'Running'\n\n    >>> print('rm1: {0}'.format(rm1))\n    rm1: Running\n    >>> print('rm2: {0}'.format(rm2))\n    rm2: Running\n\n    # When the `state` attribute is modified from instance `rm2`,\n    # the value of `state` in instance `rm1` also changes\n    >>> rm2.state = 'Zombie'\n\n    >>> print('rm1: {0}'.format(rm1))\n    rm1: Zombie\n    >>> print('rm2: {0}'.format(rm2))\n    rm2: Zombie\n\n    # Even though `rm1` and `rm2` share attributes, the instances are not the same\n    >>> rm1 is rm2\n    False\n\n    # New instances also get the same shared state\n    >>> rm3 = YourBorg()\n\n    >>> print('rm1: {0}'.format(rm1))\n    rm1: Zombie\n    >>> print('rm2: {0}'.format(rm2))\n    rm2: Zombie\n    >>> print('rm3: {0}'.format(rm3))\n    rm3: Zombie\n\n    # A new instance can explicitly change the state during creation\n    >>> rm4 = YourBorg('Running')\n\n    >>> print('rm4: {0}'.format(rm4))\n    rm4: Running\n\n    # Existing instances reflect that change as well\n    >>> print('rm3: {0}'.format(rm3))\n    rm3: Running\n    "
if __name__ == '__main__':
    import doctest
    doctest.testmod()