"""
Functionality copied from the toolz package to avoid having
to add toolz as a dependency.

See https://github.com/pytoolz/toolz/.

toolz is relased under BSD licence. Below is the licence text
from toolz as it appeared when copying the code.

--------------------------------------------------------------

Copyright (c) 2013 Matthew Rocklin

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of toolz nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""
import operator
from functools import reduce

def get_in(keys, coll, default=None, no_default=False):
    if False:
        i = 10
        return i + 15
    "\n    NB: This is a straight copy of the get_in implementation found in\n        the toolz library (https://github.com/pytoolz/toolz/). It works\n        with persistent data structures as well as the corresponding\n        datastructures from the stdlib.\n\n    Returns coll[i0][i1]...[iX] where [i0, i1, ..., iX]==keys.\n\n    If coll[i0][i1]...[iX] cannot be found, returns ``default``, unless\n    ``no_default`` is specified, then it raises KeyError or IndexError.\n\n    ``get_in`` is a generalization of ``operator.getitem`` for nested data\n    structures such as dictionaries and lists.\n    >>> from pyrsistent import freeze\n    >>> transaction = freeze({'name': 'Alice',\n    ...                       'purchase': {'items': ['Apple', 'Orange'],\n    ...                                    'costs': [0.50, 1.25]},\n    ...                       'credit card': '5555-1234-1234-1234'})\n    >>> get_in(['purchase', 'items', 0], transaction)\n    'Apple'\n    >>> get_in(['name'], transaction)\n    'Alice'\n    >>> get_in(['purchase', 'total'], transaction)\n    >>> get_in(['purchase', 'items', 'apple'], transaction)\n    >>> get_in(['purchase', 'items', 10], transaction)\n    >>> get_in(['purchase', 'total'], transaction, 0)\n    0\n    >>> get_in(['y'], {}, no_default=True)\n    Traceback (most recent call last):\n        ...\n    KeyError: 'y'\n    "
    try:
        return reduce(operator.getitem, keys, coll)
    except (KeyError, IndexError, TypeError):
        if no_default:
            raise
        return default