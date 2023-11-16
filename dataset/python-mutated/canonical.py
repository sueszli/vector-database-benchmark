"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import abc
import copy
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.utilities import performance_utils as pu
from cvxpy.utilities.deterministic import unique_list

class Canonical:
    """
    An interface for objects that can be canonicalized.
    """
    __metaclass__ = abc.ABCMeta

    @property
    def expr(self):
        if False:
            while True:
                i = 10
        if not len(self.args) == 1:
            raise ValueError("'expr' is ambiguous, there should be only one argument")
        return self.args[0]

    @pu.lazyprop
    def canonical_form(self):
        if False:
            print('Hello World!')
        'The graph implementation of the object stored as a property.\n\n        Returns:\n            A tuple of (affine expression, [constraints]).\n        '
        return self.canonicalize()

    def variables(self):
        if False:
            while True:
                i = 10
        'Returns all the variables present in the arguments.\n        '
        return unique_list([var for arg in self.args for var in arg.variables()])

    def parameters(self):
        if False:
            print('Hello World!')
        'Returns all the parameters present in the arguments.\n        '
        return unique_list([param for arg in self.args for param in arg.parameters()])

    def constants(self):
        if False:
            i = 10
            return i + 15
        'Returns all the constants present in the arguments.\n        '
        return unique_list([const for arg in self.args for const in arg.constants()])

    def tree_copy(self, id_objects=None):
        if False:
            return 10
        new_args = []
        for arg in self.args:
            if isinstance(arg, list):
                arg_list = [elem.tree_copy(id_objects) for elem in arg]
                new_args.append(arg_list)
            else:
                new_args.append(arg.tree_copy(id_objects))
        return self.copy(args=new_args, id_objects=id_objects)

    def copy(self, args=None, id_objects=None):
        if False:
            while True:
                i = 10
        'Returns a shallow copy of the object.\n\n        Used to reconstruct an object tree.\n\n        Parameters\n        ----------\n        args : list, optional\n            The arguments to reconstruct the object. If args=None, use the\n            current args of the object.\n\n        Returns\n        -------\n        Expression\n        '
        id_objects = {} if id_objects is None else id_objects
        if id(self) in id_objects:
            return id_objects[id(self)]
        if args is None:
            args = self.args
        else:
            assert len(args) == len(self.args)
        data = self.get_data()
        if data is not None:
            return type(self)(*args + data)
        else:
            return type(self)(*args)

    def __copy__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called by copy.copy()\n        Creates a shallow copy of the object, that is, the copied object refers to the same\n        leaf nodes as the original object. Non-leaf nodes are recreated.\n        Constraints keep their .id attribute, as it is used to propagate dual variables.\n\n        Summary:\n        ========\n        Leafs:              Same object\n        Constraints:        New object with same .id\n        Other expressions:  New object with new .id\n        '
        return self.copy()

    def __deepcopy__(self, memo):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called by copy.deepcopy()\n        Creates an independent copy of the object while maintaining the relationship between the\n        nodes in the expression tree.\n        '
        cvxpy_id = getattr(self, 'id', None)
        if cvxpy_id is not None and cvxpy_id in memo:
            return memo[cvxpy_id]
        else:
            with DefaultDeepCopyContextManager(self):
                new = copy.deepcopy(self, memo)
            if getattr(self, 'id', None) is not None:
                new_id = lu.get_id()
                new.id = new_id
            memo[cvxpy_id] = new
            return new

    def get_data(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Returns info needed to reconstruct the object besides the args.\n\n        Returns\n        -------\n        list\n        '
        return None

    def atoms(self):
        if False:
            i = 10
            return i + 15
        'Returns all the atoms present in the args.\n\n        Returns\n        -------\n        list\n        '
        return unique_list((atom for arg in self.args for atom in arg.atoms()))
_MISSING = object()

class DefaultDeepCopyContextManager:
    """
    override custom __deepcopy__ implementation and call copy.deepcopy's implementation instead
    """

    def __init__(self, item):
        if False:
            return 10
        self.item = item
        self.deepcopy = None

    def __enter__(self):
        if False:
            return 10
        self.deepcopy = getattr(self.item, '__deepcopy__', _MISSING)
        if self.deepcopy is not _MISSING:
            self.item.__deepcopy__ = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            while True:
                i = 10
        if self.deepcopy is not _MISSING:
            self.item.__deepcopy__ = self.deepcopy
            self.deepcopy = _MISSING