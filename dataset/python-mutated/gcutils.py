"""The Python Garbage Collector (`GC`_) doesn't usually get too much
attention, probably because:

  - Python's `reference counting`_ effectively handles the vast majority of
    unused objects
  - People are slowly learning to avoid implementing `object.__del__()`_
  - The collection itself strikes a good balance between simplicity and
    power (`tunable generation sizes`_)
  - The collector itself is fast and rarely the cause of long pauses
    associated with GC in other runtimes

Even so, for many applications, the time will come when the developer
will need to track down:

  - Circular references
  - Misbehaving objects (locks, ``__del__()``)
  - Memory leaks
  - Or just ways to shave off a couple percent of execution time

Thanks to the :mod:`gc` module, the GC is a well-instrumented entry
point for exactly these tasks, and ``gcutils`` aims to facilitate it
further.

.. _GC: https://docs.python.org/2/glossary.html#term-garbage-collection
.. _reference counting: https://docs.python.org/2/glossary.html#term-reference-count
.. _object.__del__(): https://docs.python.org/2/glossary.html#term-reference-count
.. _tunable generation sizes: https://docs.python.org/2/library/gc.html#gc.set_threshold
"""
from __future__ import print_function
import gc
import sys
__all__ = ['get_all', 'GCToggler', 'toggle_gc', 'toggle_gc_postcollect']

def get_all(type_obj, include_subtypes=True):
    if False:
        while True:
            i = 10
    "Get a list containing all instances of a given type.  This will\n    work for the vast majority of types out there.\n\n    >>> class Ratking(object): pass\n    >>> wiki, hak, sport = Ratking(), Ratking(), Ratking()\n    >>> len(get_all(Ratking))\n    3\n\n    However, there are some exceptions. For example, ``get_all(bool)``\n    returns an empty list because ``True`` and ``False`` are\n    themselves built-in and not tracked.\n\n    >>> get_all(bool)\n    []\n\n    Still, it's not hard to see how this functionality can be used to\n    find all instances of a leaking type and track them down further\n    using :func:`gc.get_referrers` and :func:`gc.get_referents`.\n\n    ``get_all()`` is optimized such that getting instances of\n    user-created types is quite fast. Setting *include_subtypes* to\n    ``False`` will further increase performance in cases where\n    instances of subtypes aren't required.\n\n    .. note::\n\n      There are no guarantees about the state of objects returned by\n      ``get_all()``, especially in concurrent environments. For\n      instance, it is possible for an object to be in the middle of\n      executing its ``__init__()`` and be only partially constructed.\n    "
    if not isinstance(type_obj, type):
        raise TypeError('expected a type, not %r' % type_obj)
    try:
        type_is_tracked = gc.is_tracked(type_obj)
    except AttributeError:
        type_is_tracked = False
    if type_is_tracked:
        to_check = gc.get_referrers(type_obj)
    else:
        to_check = gc.get_objects()
    if include_subtypes:
        ret = [x for x in to_check if isinstance(x, type_obj)]
    else:
        ret = [x for x in to_check if type(x) is type_obj]
    return ret
_IS_PYPY = '__pypy__' in sys.builtin_module_names
if _IS_PYPY:
    del get_all

class GCToggler(object):
    """The ``GCToggler`` is a context-manager that allows one to safely
    take more control of your garbage collection schedule. Anecdotal
    experience says certain object-creation-heavy tasks see speedups
    of around 10% by simply doing one explicit collection at the very
    end, especially if most of the objects will stay resident.

    Two GCTogglers are already present in the ``gcutils`` module:

    - :data:`toggle_gc` simply turns off GC at context entrance, and
      re-enables at exit
    - :data:`toggle_gc_postcollect` does the same, but triggers an
      explicit collection after re-enabling.

    >>> with toggle_gc:
    ...     x = [object() for i in range(1000)]

    Between those two instances, the ``GCToggler`` type probably won't
    be used much directly, but is documented for inheritance purposes.
    """

    def __init__(self, postcollect=False):
        if False:
            print('Hello World!')
        self.postcollect = postcollect

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        gc.disable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            while True:
                i = 10
        gc.enable()
        if self.postcollect:
            gc.collect()
toggle_gc = GCToggler()
'A context manager for disabling GC for a code block. See\n:class:`GCToggler` for more details.'
toggle_gc_postcollect = GCToggler(postcollect=True)
'A context manager for disabling GC for a code block, and collecting\nbefore re-enabling. See :class:`GCToggler` for more details.'