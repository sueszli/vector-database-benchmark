"""
This module can be used to solve problems related
to 2D Cables.
"""
from sympy.core.sympify import sympify
from sympy.core.symbol import Symbol

class Cable:
    """
    Cables are structures in engineering that support
    the applied transverse loads through the tensile
    resistance developed in its members.

    Cables are widely used in suspension bridges, tension
    leg offshore platforms, transmission lines, and find
    use in several other engineering applications.

    Examples
    ========
    A cable is supported at (0, 10) and (10, 10). Two point loads
    acting vertically downwards act on the cable, one with magnitude 3 kN
    and acting 2 meters from the left support and 3 meters below it, while
    the other with magnitude 2 kN is 6 meters from the left support and
    6 meters below it.

    >>> from sympy.physics.continuum_mechanics.cable import Cable
    >>> c = Cable(('A', 0, 10), ('B', 10, 10))
    >>> c.apply_load(-1, ('P', 2, 7, 3, 270))
    >>> c.apply_load(-1, ('Q', 6, 4, 2, 270))
    >>> c.loads
    {'distributed': {}, 'point_load': {'P': [3, 270], 'Q': [2, 270]}}
    >>> c.loads_position
    {'P': [2, 7], 'Q': [6, 4]}
    """

    def __init__(self, support_1, support_2):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initializes the class.\n\n        Parameters\n        ==========\n\n        support_1 and support_2 are tuples of the form\n        (label, x, y), where\n\n        label : String or symbol\n            The label of the support\n\n        x : Sympifyable\n            The x coordinate of the position of the support\n\n        y : Sympifyable\n            The y coordinate of the position of the support\n        '
        self._left_support = []
        self._right_support = []
        self._supports = {}
        self._support_labels = []
        self._loads = {'distributed': {}, 'point_load': {}}
        self._loads_position = {}
        self._length = 0
        self._reaction_loads = {}
        if support_1[0] == support_2[0]:
            raise ValueError('Supports can not have the same label')
        elif support_1[1] == support_2[1]:
            raise ValueError('Supports can not be at the same location')
        x1 = sympify(support_1[1])
        y1 = sympify(support_1[2])
        self._supports[support_1[0]] = [x1, y1]
        x2 = sympify(support_2[1])
        y2 = sympify(support_2[2])
        self._supports[support_2[0]] = [x2, y2]
        if support_1[1] < support_2[1]:
            self._left_support.append(x1)
            self._left_support.append(y1)
            self._right_support.append(x2)
            self._right_support.append(y2)
            self._support_labels.append(support_1[0])
            self._support_labels.append(support_2[0])
        else:
            self._left_support.append(x2)
            self._left_support.append(y2)
            self._right_support.append(x1)
            self._right_support.append(y1)
            self._support_labels.append(support_2[0])
            self._support_labels.append(support_1[0])
        for i in self._support_labels:
            self._reaction_loads[Symbol('R_' + i + '_x')] = 0
            self._reaction_loads[Symbol('R_' + i + '_y')] = 0

    @property
    def supports(self):
        if False:
            print('Hello World!')
        '\n        Returns the supports of the cable along with their\n        positions.\n        '
        return self._supports

    @property
    def left_support(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the position of the left support.\n        '
        return self._left_support

    @property
    def right_support(self):
        if False:
            while True:
                i = 10
        '\n        Returns the position of the right support.\n        '
        return self._right_support

    @property
    def loads(self):
        if False:
            while True:
                i = 10
        '\n        Returns the magnitude and direction of the loads\n        acting on the cable.\n        '
        return self._loads

    @property
    def loads_position(self):
        if False:
            while True:
                i = 10
        '\n        Returns the position of the point loads acting on the\n        cable.\n        '
        return self._loads_position

    @property
    def length(self):
        if False:
            print('Hello World!')
        '\n        Returns the length of the cable.\n        '
        return self._length

    @property
    def reaction_loads(self):
        if False:
            print('Hello World!')
        '\n        Returns the reaction forces at the supports, which are\n        initialized to 0.\n        '
        return self._reaction_loads

    def apply_length(self, length):
        if False:
            print('Hello World!')
        "\n        This method specifies the length of the cable\n\n        Parameters\n        ==========\n\n        length : Sympifyable\n            The length of the cable\n\n        Examples\n        ========\n\n        >>> from sympy.physics.continuum_mechanics.cable import Cable\n        >>> c = Cable(('A', 0, 10), ('B', 10, 10))\n        >>> c.apply_length(20)\n        >>> c.length\n        20\n        "
        dist = ((self._left_support[0] - self._right_support[0]) ** 2 - (self._left_support[1] - self._right_support[1]) ** 2) ** (1 / 2)
        if length < dist:
            raise ValueError('length should not be less than the distance between the supports')
        self._length = length

    def change_support(self, label, new_support):
        if False:
            while True:
                i = 10
        "\n        This method changes the mentioned support with a new support.\n\n        Parameters\n        ==========\n        label: String or symbol\n            The label of the support to be changed\n\n        new_support: Tuple of the form (new_label, x, y)\n            new_label: String or symbol\n                The label of the new support\n\n            x: Sympifyable\n                The x-coordinate of the position of the new support.\n\n            y: Sympifyable\n                The y-coordinate of the position of the new support.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.continuum_mechanics.cable import Cable\n        >>> c = Cable(('A', 0, 10), ('B', 10, 10))\n        >>> c.supports\n        {'A': [0, 10], 'B': [10, 10]}\n        >>> c.change_support('B', ('C', 5, 6))\n        >>> c.supports\n        {'A': [0, 10], 'C': [5, 6]}\n        "
        if label not in self._supports:
            raise ValueError('No support exists with the given label')
        i = self._support_labels.index(label)
        rem_label = self._support_labels[(i + 1) % 2]
        x1 = self._supports[rem_label][0]
        y1 = self._supports[rem_label][1]
        x = sympify(new_support[1])
        y = sympify(new_support[2])
        for l in self._loads_position:
            if l[0] >= max(x, x1) or l[0] <= min(x, x1):
                raise ValueError('The change in support will throw an existing load out of range')
        self._supports.pop(label)
        self._left_support.clear()
        self._right_support.clear()
        self._reaction_loads.clear()
        self._support_labels.remove(label)
        self._supports[new_support[0]] = [x, y]
        if x1 < x:
            self._left_support.append(x1)
            self._left_support.append(y1)
            self._right_support.append(x)
            self._right_support.append(y)
            self._support_labels.append(new_support[0])
        else:
            self._left_support.append(x)
            self._left_support.append(y)
            self._right_support.append(x1)
            self._right_support.append(y1)
            self._support_labels.insert(0, new_support[0])
        for i in self._support_labels:
            self._reaction_loads[Symbol('R_' + i + '_x')] = 0
            self._reaction_loads[Symbol('R_' + i + '_y')] = 0

    def apply_load(self, order, load):
        if False:
            return 10
        "\n        This method adds load to the cable.\n\n        Parameters\n        ==========\n\n        order : Integer\n            The order of the applied load.\n\n                - For point loads, order = -1\n                - For distributed load, order = 0\n\n        load : tuple\n\n            * For point loads, load is of the form (label, x, y, magnitude, direction), where:\n\n            label : String or symbol\n                The label of the load\n\n            x : Sympifyable\n                The x coordinate of the position of the load\n\n            y : Sympifyable\n                The y coordinate of the position of the load\n\n            magnitude : Sympifyable\n                The magnitude of the load. It must always be positive\n\n            direction : Sympifyable\n                The angle, in degrees, that the load vector makes with the horizontal\n                in the counter-clockwise direction. It takes the values 0 to 360,\n                inclusive.\n\n\n            * For uniformly distributed load, load is of the form (label, magnitude)\n\n            label : String or symbol\n                The label of the load\n\n            magnitude : Sympifyable\n                The magnitude of the load. It must always be positive\n\n        Examples\n        ========\n\n        For a point load of magnitude 12 units inclined at 30 degrees with the horizontal:\n\n        >>> from sympy.physics.continuum_mechanics.cable import Cable\n        >>> c = Cable(('A', 0, 10), ('B', 10, 10))\n        >>> c.apply_load(-1, ('Z', 5, 5, 12, 30))\n        >>> c.loads\n        {'distributed': {}, 'point_load': {'Z': [12, 30]}}\n        >>> c.loads_position\n        {'Z': [5, 5]}\n\n\n        For a uniformly distributed load of magnitude 9 units:\n\n        >>> from sympy.physics.continuum_mechanics.cable import Cable\n        >>> c = Cable(('A', 0, 10), ('B', 10, 10))\n        >>> c.apply_load(0, ('X', 9))\n        >>> c.loads\n        {'distributed': {'X': 9}, 'point_load': {}}\n        "
        if order == -1:
            if len(self._loads['distributed']) != 0:
                raise ValueError('Distributed load already exists')
            label = load[0]
            if label in self._loads['point_load']:
                raise ValueError('Label already exists')
            x = sympify(load[1])
            y = sympify(load[2])
            if x > self._right_support[0] or x < self._left_support[0]:
                raise ValueError('The load should be positioned between the supports')
            magnitude = sympify(load[3])
            direction = sympify(load[4])
            self._loads['point_load'][label] = [magnitude, direction]
            self._loads_position[label] = [x, y]
        elif order == 0:
            if len(self._loads_position) != 0:
                raise ValueError('Point load(s) already exist')
            label = load[0]
            if label in self._loads['distributed']:
                raise ValueError('Label already exists')
            magnitude = sympify(load[1])
            self._loads['distributed'][label] = magnitude
        else:
            raise ValueError('Order should be either -1 or 0')

    def remove_loads(self, *args):
        if False:
            while True:
                i = 10
        "\n        This methods removes the specified loads.\n\n        Parameters\n        ==========\n        This input takes multiple label(s) as input\n        label(s): String or symbol\n            The label(s) of the loads to be removed.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.continuum_mechanics.cable import Cable\n        >>> c = Cable(('A', 0, 10), ('B', 10, 10))\n        >>> c.apply_load(-1, ('Z', 5, 5, 12, 30))\n        >>> c.loads\n        {'distributed': {}, 'point_load': {'Z': [12, 30]}}\n        >>> c.remove_loads('Z')\n        >>> c.loads\n        {'distributed': {}, 'point_load': {}}\n        "
        for i in args:
            if len(self._loads_position) == 0:
                if i not in self._loads['distributed']:
                    raise ValueError('Error removing load ' + i + ': no such load exists')
                else:
                    self._loads['disrtibuted'].pop(i)
            elif i not in self._loads['point_load']:
                raise ValueError('Error removing load ' + i + ': no such load exists')
            else:
                self._loads['point_load'].pop(i)
                self._loads_position.pop(i)