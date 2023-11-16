from itertools import chain
from coalib.core.Graphs import traverse_graph

class DependencyTracker:
    """
    A ``DependencyTracker`` allows to register and manage dependencies between
    objects.

    This class uses a directed graph to track relations.

    Add a dependency relation between two objects:

    >>> object1 = object()
    >>> object2 = object()
    >>> tracker = DependencyTracker()
    >>> tracker.add(object2, object1)

    This would define that ``object1`` is dependent on ``object2``.

    If you define that ``object2`` has its dependency duty fulfilled, you can
    resolve it:

    >>> resolved = tracker.resolve(object2)
    >>> resolved
    {<object object at ...>}
    >>> resolved_object = resolved.pop()
    >>> resolved_object is object1
    True

    This returns all objects that are now freed, meaning they have no
    dependencies any more.

    >>> object3 = object()
    >>> tracker.add(object2, object1)
    >>> tracker.add(object3, object1)
    >>> tracker.resolve(object2)
    set()
    >>> tracker.resolve(object3)
    {<object object at ...>}

    The ones who instantiate a ``DependencyTracker`` are responsible for
    resolving dependencies in the right order. Dependencies which are itself
    dependent will be forcefully resolved and removed from their according
    dependencies too.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self._dependency_dict = {}

    def get_dependants(self, dependency):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns all immediate dependants for the given dependency.\n\n        >>> tracker = DependencyTracker()\n        >>> tracker.add(0, 1)\n        >>> tracker.add(0, 2)\n        >>> tracker.add(1, 3)\n        >>> tracker.get_dependants(0)\n        {1, 2}\n        >>> tracker.get_dependants(1)\n        {3}\n        >>> tracker.get_dependants(2)\n        set()\n\n        :param dependency:\n            The dependency to retrieve all dependants from.\n        :return:\n            A set of dependants.\n        '
        try:
            return set(self._dependency_dict[dependency])
        except KeyError:
            return set()

    def get_dependencies(self, dependant):
        if False:
            i = 10
            return i + 15
        '\n        Returns all immediate dependencies of a given dependant.\n\n        >>> tracker = DependencyTracker()\n        >>> tracker.add(0, 1)\n        >>> tracker.add(0, 2)\n        >>> tracker.add(1, 2)\n        >>> tracker.get_dependencies(0)\n        set()\n        >>> tracker.get_dependencies(1)\n        {0}\n        >>> tracker.get_dependencies(2)\n        {0, 1}\n\n        :param dependant:\n            The dependant to retrieve all dependencies from.\n        :return:\n            A set of dependencies.\n        '
        return set((dependency for (dependency, dependants) in self._dependency_dict.items() if dependant in dependants))

    def get_all_dependants(self, dependency):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a set of all dependants of the given dependency, even\n        indirectly related ones.\n\n        >>> tracker = DependencyTracker()\n        >>> tracker.add(0, 1)\n        >>> tracker.add(1, 2)\n        >>> tracker.get_all_dependants(0)\n        {1, 2}\n\n        :param dependency:\n            The dependency to get all dependants for.\n        :return:\n            A set of dependants.\n        '
        dependants = set()

        def append_to_dependants(prev, nxt):
            if False:
                while True:
                    i = 10
            dependants.add(nxt)
        traverse_graph([dependency], lambda node: self._dependency_dict.get(node, frozenset()), append_to_dependants)
        return dependants

    def get_all_dependencies(self, dependant):
        if False:
            print('Hello World!')
        '\n        Returns a set of all dependencies of the given dependants, even\n        indirectly related ones.\n\n        >>> tracker = DependencyTracker()\n        >>> tracker.add(0, 1)\n        >>> tracker.add(1, 2)\n        >>> tracker.get_all_dependencies(2)\n        {0, 1}\n\n        :param dependant:\n            The dependant to get all dependencies for.\n        :return:\n            A set of dependencies.\n        '
        dependencies = set()

        def append_to_dependencies(prev, nxt):
            if False:
                for i in range(10):
                    print('nop')
            dependencies.add(nxt)
        traverse_graph([dependant], lambda node: {dependency for (dependency, dependants) in self._dependency_dict.items() if node in dependants}, append_to_dependencies)
        return dependencies

    @property
    def dependants(self):
        if False:
            while True:
                i = 10
        '\n        Returns a set of all registered dependants.\n\n        >>> tracker = DependencyTracker()\n        >>> tracker.add(0, 1)\n        >>> tracker.add(0, 2)\n        >>> tracker.add(1, 3)\n        >>> tracker.dependants\n        {1, 2, 3}\n        '
        return set(chain.from_iterable(self._dependency_dict.values()))

    @property
    def dependencies(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a set of all registered dependencies.\n\n        >>> tracker = DependencyTracker()\n        >>> tracker.add(0, 1)\n        >>> tracker.add(0, 2)\n        >>> tracker.add(1, 3)\n        >>> tracker.dependencies\n        {0, 1}\n        '
        return set(self._dependency_dict.keys())

    def __iter__(self):
        if False:
            print('Hello World!')
        "\n        Returns an iterator that iterates over all dependency relations.\n\n        >>> tracker = DependencyTracker()\n        >>> tracker.add(0, 1)\n        >>> tracker.add(0, 2)\n        >>> tracker.add(1, 2)\n        >>> for dependency, dependant in sorted(tracker):\n        ...     print(dependency, '->', dependant)\n        0 -> 1\n        0 -> 2\n        1 -> 2\n        "
        return ((dependency, dependant) for (dependency, dependants) in self._dependency_dict.items() for dependant in dependants)

    def add(self, dependency, dependant):
        if False:
            return 10
        '\n        Add a dependency relation.\n\n        This function does not check for circular dependencies.\n\n        >>> tracker = DependencyTracker()\n        >>> tracker.add(0, 1)\n        >>> tracker.add(0, 2)\n        >>> tracker.resolve(0)\n        {1, 2}\n\n        :param dependency:\n            The object that is the dependency.\n        :param dependant:\n            The object that is the dependant.\n        '
        if dependency not in self._dependency_dict:
            self._dependency_dict[dependency] = set()
        self._dependency_dict[dependency].add(dependant)

    def resolve(self, dependency):
        if False:
            for i in range(10):
                print('nop')
        '\n        Resolves all dependency-relations from the given dependency, and frees\n        and returns dependants with no more dependencies. If the given\n        dependency is itself a dependant, all those relations are also removed.\n\n        >>> tracker = DependencyTracker()\n        >>> tracker.add(0, 1)\n        >>> tracker.add(0, 2)\n        >>> tracker.add(2, 3)\n        >>> tracker.resolve(0)\n        {1, 2}\n        >>> tracker.resolve(2)\n        {3}\n        >>> tracker.resolve(2)\n        set()\n\n        :param dependency:\n            The dependency.\n        :return:\n            Returns a set of dependants whose dependencies were all resolved.\n        '
        dependencies_to_remove = []
        for (tracked_dependency, dependants) in self._dependency_dict.items():
            if dependency in dependants:
                dependants.remove(dependency)
                if not dependants:
                    dependencies_to_remove.append(tracked_dependency)
        for tracked_dependency in dependencies_to_remove:
            del self._dependency_dict[tracked_dependency]
        possible_freed_dependants = self._dependency_dict.pop(dependency, set())
        non_free_dependants = set()
        for possible_freed_dependant in possible_freed_dependants:
            for dependants in self._dependency_dict.values():
                if possible_freed_dependant in dependants:
                    non_free_dependants.add(possible_freed_dependant)
                    break
        return possible_freed_dependants - non_free_dependants

    def check_circular_dependencies(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks whether there are circular dependency conflicts.\n\n        >>> tracker = DependencyTracker()\n        >>> tracker.add(0, 1)\n        >>> tracker.add(1, 0)\n        >>> tracker.check_circular_dependencies()\n        Traceback (most recent call last):\n         ...\n        coalib.core.CircularDependencyError.CircularDependencyError: ...\n\n        :raises CircularDependencyError:\n            Raised on circular dependency conflicts.\n        '
        traverse_graph(self._dependency_dict.keys(), lambda node: self._dependency_dict.get(node, frozenset()))

    @property
    def are_dependencies_resolved(self):
        if False:
            while True:
                i = 10
        '\n        Checks whether all dependencies in this ``DependencyTracker`` instance\n        are resolved.\n\n        >>> tracker = DependencyTracker()\n        >>> tracker.are_dependencies_resolved\n        True\n        >>> tracker.add(0, 1)\n        >>> tracker.are_dependencies_resolved\n        False\n        >>> tracker.resolve(0)\n        {1}\n        >>> tracker.are_dependencies_resolved\n        True\n\n        :return:\n            ``True`` when all dependencies resolved, ``False`` if not.\n        '
        return not self._dependency_dict