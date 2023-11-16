"""Implementations of actuators for linked force and torque application."""
from abc import ABC, abstractmethod
from sympy import S, sympify
from sympy.physics.mechanics import PathwayBase, PinJoint, ReferenceFrame, RigidBody, Torque, Vector
__all__ = ['ActuatorBase', 'ForceActuator', 'LinearDamper', 'LinearSpring', 'TorqueActuator']

class ActuatorBase(ABC):
    """Abstract base class for all actuator classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom actuator types through subclassing.

    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Initializer for ``ActuatorBase``.'
        pass

    @abstractmethod
    def to_loads(self):
        if False:
            print('Hello World!')
        'Loads required by the equations of motion method classes.\n\n        Explanation\n        ===========\n\n        ``KanesMethod`` requires a list of ``Point``-``Vector`` tuples to be\n        passed to the ``loads`` parameters of its ``kanes_equations`` method\n        when constructing the equations of motion. This method acts as a\n        utility to produce the correctly-structred pairs of points and vectors\n        required so that these can be easily concatenated with other items in\n        the list of loads and passed to ``KanesMethod.kanes_equations``. These\n        loads are also in the correct form to also be passed to the other\n        equations of motion method classes, e.g. ``LagrangesMethod``.\n\n        '
        pass

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Default representation of an actuator.'
        return f'{self.__class__.__name__}()'

class ForceActuator(ActuatorBase):
    """Force-producing actuator.

    Explanation
    ===========

    A ``ForceActuator`` is an actuator that produces a (expansile) force along
    its length.

    A force actuator uses a pathway instance to determine the direction and
    number of forces that it applies to a system. Consider the simplest case
    where a ``LinearPathway`` instance is used. This pathway is made up of two
    points that can move relative to each other, and results in a pair of equal
    and opposite forces acting on the endpoints. If the positive time-varying
    Euclidean distance between the two points is defined, then the "extension
    velocity" is the time derivative of this distance. The extension velocity
    is positive when the two points are moving away from each other and
    negative when moving closer to each other. The direction for the force
    acting on either point is determined by constructing a unit vector directed
    from the other point to this point. This establishes a sign convention such
    that a positive force magnitude tends to push the points apart, this is the
    meaning of "expansile" in this context. The following diagram shows the
    positive force sense and the distance between the points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    To construct an actuator, an expression (or symbol) must be supplied to
    represent the force it can produce, alongside a pathway specifying its line
    of action. Let's also create a global reference frame and spatially fix one
    of the points in it while setting the other to be positioned such that it
    can freely move in the frame's x direction specified by the coordinate
    ``q``.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (ForceActuator, LinearPathway,
    ...     Point, ReferenceFrame)
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> force = symbols('F')
    >>> pA, pB = Point('pA'), Point('pB')
    >>> pA.set_vel(N, 0)
    >>> pB.set_pos(pA, q*N.x)
    >>> pB.pos_from(pA)
    q(t)*N.x
    >>> linear_pathway = LinearPathway(pA, pB)
    >>> actuator = ForceActuator(force, linear_pathway)
    >>> actuator
    ForceActuator(F, LinearPathway(pA, pB))

    Parameters
    ==========

    force : Expr
        The scalar expression defining the (expansile) force that the actuator
        produces.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.

    """

    def __init__(self, force, pathway):
        if False:
            i = 10
            return i + 15
        'Initializer for ``ForceActuator``.\n\n        Parameters\n        ==========\n\n        force : Expr\n            The scalar expression defining the (expansile) force that the\n            actuator produces.\n        pathway : PathwayBase\n            The pathway that the actuator follows. This must be an instance of\n            a concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.\n\n        '
        self.force = force
        self.pathway = pathway

    @property
    def force(self):
        if False:
            while True:
                i = 10
        'The magnitude of the force produced by the actuator.'
        return self._force

    @force.setter
    def force(self, force):
        if False:
            while True:
                i = 10
        if hasattr(self, '_force'):
            msg = f"Can't set attribute `force` to {repr(force)} as it is immutable."
            raise AttributeError(msg)
        self._force = sympify(force, strict=True)

    @property
    def pathway(self):
        if False:
            for i in range(10):
                print('nop')
        "The ``Pathway`` defining the actuator's line of action."
        return self._pathway

    @pathway.setter
    def pathway(self, pathway):
        if False:
            print('Hello World!')
        if hasattr(self, '_pathway'):
            msg = f"Can't set attribute `pathway` to {repr(pathway)} as it is immutable."
            raise AttributeError(msg)
        if not isinstance(pathway, PathwayBase):
            msg = f'Value {repr(pathway)} passed to `pathway` was of type {type(pathway)}, must be {PathwayBase}.'
            raise TypeError(msg)
        self._pathway = pathway

    def to_loads(self):
        if False:
            while True:
                i = 10
        "Loads required by the equations of motion method classes.\n\n        Explanation\n        ===========\n\n        ``KanesMethod`` requires a list of ``Point``-``Vector`` tuples to be\n        passed to the ``loads`` parameters of its ``kanes_equations`` method\n        when constructing the equations of motion. This method acts as a\n        utility to produce the correctly-structred pairs of points and vectors\n        required so that these can be easily concatenated with other items in\n        the list of loads and passed to ``KanesMethod.kanes_equations``. These\n        loads are also in the correct form to also be passed to the other\n        equations of motion method classes, e.g. ``LagrangesMethod``.\n\n        Examples\n        ========\n\n        The below example shows how to generate the loads produced by a force\n        actuator that follows a linear pathway. In this example we'll assume\n        that the force actuator is being used to model a simple linear spring.\n        First, create a linear pathway between two points separated by the\n        coordinate ``q`` in the ``x`` direction of the global frame ``N``.\n\n        >>> from sympy.physics.mechanics import (LinearPathway, Point,\n        ...     ReferenceFrame)\n        >>> from sympy.physics.vector import dynamicsymbols\n        >>> q = dynamicsymbols('q')\n        >>> N = ReferenceFrame('N')\n        >>> pA, pB = Point('pA'), Point('pB')\n        >>> pB.set_pos(pA, q*N.x)\n        >>> pathway = LinearPathway(pA, pB)\n\n        Now create a symbol ``k`` to describe the spring's stiffness and\n        instantiate a force actuator that produces a (contractile) force\n        proportional to both the spring's stiffness and the pathway's length.\n        Note that actuator classes use the sign convention that expansile\n        forces are positive, so for a spring to produce a contractile force the\n        spring force needs to be calculated as the negative for the stiffness\n        multiplied by the length.\n\n        >>> from sympy import symbols\n        >>> from sympy.physics.mechanics import ForceActuator\n        >>> stiffness = symbols('k')\n        >>> spring_force = -stiffness*pathway.length\n        >>> spring = ForceActuator(spring_force, pathway)\n\n        The forces produced by the spring can be generated in the list of loads\n        form that ``KanesMethod`` (and other equations of motion methods)\n        requires by calling the ``to_loads`` method.\n\n        >>> spring.to_loads()\n        [(pA, k*q(t)*N.x), (pB, - k*q(t)*N.x)]\n\n        A simple linear damper can be modeled in a similar way. Create another\n        symbol ``c`` to describe the dampers damping coefficient. This time\n        instantiate a force actuator that produces a force proportional to both\n        the damper's damping coefficient and the pathway's extension velocity.\n        Note that the damping force is negative as it acts in the opposite\n        direction to which the damper is changing in length.\n\n        >>> damping_coefficient = symbols('c')\n        >>> damping_force = -damping_coefficient*pathway.extension_velocity\n        >>> damper = ForceActuator(damping_force, pathway)\n\n        Again, the forces produces by the damper can be generated by calling\n        the ``to_loads`` method.\n\n        >>> damper.to_loads()\n        [(pA, c*Derivative(q(t), t)*N.x), (pB, - c*Derivative(q(t), t)*N.x)]\n\n        "
        return self.pathway.to_loads(self.force)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Representation of a ``ForceActuator``.'
        return f'{self.__class__.__name__}({self.force}, {self.pathway})'

class LinearSpring(ForceActuator):
    """A spring with its spring force as a linear function of its length.

    Explanation
    ===========

    Note that the "linear" in the name ``LinearSpring`` refers to the fact that
    the spring force is a linear function of the springs length. I.e. for a
    linear spring with stiffness ``k``, distance between its ends of ``x``, and
    an equilibrium length of ``0``, the spring force will be ``-k*x``, which is
    a linear function in ``x``. To create a spring that follows a linear, or
    straight, pathway between its two ends, a ``LinearPathway`` instance needs
    to be passed to the ``pathway`` parameter.

    A ``LinearSpring`` is a subclass of ``ForceActuator`` and so follows the
    same sign conventions for length, extension velocity, and the direction of
    the forces it applies to its points of attachment on bodies. The sign
    convention for the direction of forces is such that, for the case where a
    linear spring is instantiated with a ``LinearPathway`` instance as its
    pathway, they act to push the two ends of the spring away from one another.
    Because springs produces a contractile force and acts to pull the two ends
    together towards the equilibrium length when stretched, the scalar portion
    of the forces on the endpoint are negative in order to flip the sign of the
    forces on the endpoints when converted into vector quantities. The
    following diagram shows the positive force sense and the distance between
    the points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    To construct a linear spring, an expression (or symbol) must be supplied to
    represent the stiffness (spring constant) of the spring, alongside a
    pathway specifying its line of action. Let's also create a global reference
    frame and spatially fix one of the points in it while setting the other to
    be positioned such that it can freely move in the frame's x direction
    specified by the coordinate ``q``.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (LinearPathway, LinearSpring,
    ...     Point, ReferenceFrame)
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> stiffness = symbols('k')
    >>> pA, pB = Point('pA'), Point('pB')
    >>> pA.set_vel(N, 0)
    >>> pB.set_pos(pA, q*N.x)
    >>> pB.pos_from(pA)
    q(t)*N.x
    >>> linear_pathway = LinearPathway(pA, pB)
    >>> spring = LinearSpring(stiffness, linear_pathway)
    >>> spring
    LinearSpring(k, LinearPathway(pA, pB))

    This spring will produce a force that is proportional to both its stiffness
    and the pathway's length. Note that this force is negative as SymPy's sign
    convention for actuators is that negative forces are contractile.

    >>> spring.force
    -k*sqrt(q(t)**2)

    To create a linear spring with a non-zero equilibrium length, an expression
    (or symbol) can be passed to the ``equilibrium_length`` parameter on
    construction on a ``LinearSpring`` instance. Let's create a symbol ``l``
    to denote a non-zero equilibrium length and create another linear spring.

    >>> l = symbols('l')
    >>> spring = LinearSpring(stiffness, linear_pathway, equilibrium_length=l)
    >>> spring
    LinearSpring(k, LinearPathway(pA, pB), equilibrium_length=l)

    The spring force of this new spring is again proportional to both its
    stiffness and the pathway's length. However, the spring will not produce
    any force when ``q(t)`` equals ``l``. Note that the force will become
    expansile when ``q(t)`` is less than ``l``, as expected.

    >>> spring.force
    -k*(-l + sqrt(q(t)**2))

    Parameters
    ==========

    stiffness : Expr
        The spring constant.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
    equilibrium_length : Expr, optional
        The length at which the spring is in equilibrium, i.e. it produces no
        force. The default value is 0, i.e. the spring force is a linear
        function of the pathway's length with no constant offset.

    See Also
    ========

    ForceActuator: force-producing actuator (superclass of ``LinearSpring``).
    LinearPathway: straight-line pathway between a pair of points.

    """

    def __init__(self, stiffness, pathway, equilibrium_length=S.Zero):
        if False:
            print('Hello World!')
        "Initializer for ``LinearSpring``.\n\n        Parameters\n        ==========\n\n        stiffness : Expr\n            The spring constant.\n        pathway : PathwayBase\n            The pathway that the actuator follows. This must be an instance of\n            a concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.\n        equilibrium_length : Expr, optional\n            The length at which the spring is in equilibrium, i.e. it produces\n            no force. The default value is 0, i.e. the spring force is a linear\n            function of the pathway's length with no constant offset.\n\n        "
        self.stiffness = stiffness
        self.pathway = pathway
        self.equilibrium_length = equilibrium_length

    @property
    def force(self):
        if False:
            for i in range(10):
                print('nop')
        'The spring force produced by the linear spring.'
        return -self.stiffness * (self.pathway.length - self.equilibrium_length)

    @force.setter
    def force(self, force):
        if False:
            return 10
        raise AttributeError("Can't set computed attribute `force`.")

    @property
    def stiffness(self):
        if False:
            for i in range(10):
                print('nop')
        'The spring constant for the linear spring.'
        return self._stiffness

    @stiffness.setter
    def stiffness(self, stiffness):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, '_stiffness'):
            msg = f"Can't set attribute `stiffness` to {repr(stiffness)} as it is immutable."
            raise AttributeError(msg)
        self._stiffness = sympify(stiffness, strict=True)

    @property
    def equilibrium_length(self):
        if False:
            print('Hello World!')
        'The length of the spring at which it produces no force.'
        return self._equilibrium_length

    @equilibrium_length.setter
    def equilibrium_length(self, equilibrium_length):
        if False:
            i = 10
            return i + 15
        if hasattr(self, '_equilibrium_length'):
            msg = f"Can't set attribute `equilibrium_length` to {repr(equilibrium_length)} as it is immutable."
            raise AttributeError(msg)
        self._equilibrium_length = sympify(equilibrium_length, strict=True)

    def __repr__(self):
        if False:
            return 10
        'Representation of a ``LinearSpring``.'
        string = f'{self.__class__.__name__}({self.stiffness}, {self.pathway}'
        if self.equilibrium_length == S.Zero:
            string += ')'
        else:
            string += f', equilibrium_length={self.equilibrium_length})'
        return string

class LinearDamper(ForceActuator):
    """A damper whose force is a linear function of its extension velocity.

    Explanation
    ===========

    Note that the "linear" in the name ``LinearDamper`` refers to the fact that
    the damping force is a linear function of the damper's rate of change in
    its length. I.e. for a linear damper with damping ``c`` and extension
    velocity ``v``, the damping force will be ``-c*v``, which is a linear
    function in ``v``. To create a damper that follows a linear, or straight,
    pathway between its two ends, a ``LinearPathway`` instance needs to be
    passed to the ``pathway`` parameter.

    A ``LinearDamper`` is a subclass of ``ForceActuator`` and so follows the
    same sign conventions for length, extension velocity, and the direction of
    the forces it applies to its points of attachment on bodies. The sign
    convention for the direction of forces is such that, for the case where a
    linear damper is instantiated with a ``LinearPathway`` instance as its
    pathway, they act to push the two ends of the damper away from one another.
    Because dampers produce a force that opposes the direction of change in
    length, when extension velocity is positive the scalar portions of the
    forces applied at the two endpoints are negative in order to flip the sign
    of the forces on the endpoints wen converted into vector quantities. When
    extension velocity is negative (i.e. when the damper is shortening), the
    scalar portions of the fofces applied are also negative so that the signs
    cancel producing forces on the endpoints that are in the same direction as
    the positive sign convention for the forces at the endpoints of the pathway
    (i.e. they act to push the endpoints away from one another). The following
    diagram shows the positive force sense and the distance between the
    points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    To construct a linear damper, an expression (or symbol) must be supplied to
    represent the damping coefficient of the damper (we'll use the symbol
    ``c``), alongside a pathway specifying its line of action. Let's also
    create a global reference frame and spatially fix one of the points in it
    while setting the other to be positioned such that it can freely move in
    the frame's x direction specified by the coordinate ``q``. The velocity
    that the two points move away from one another can be specified by the
    coordinate ``u`` where ``u`` is the first time derivative of ``q``
    (i.e., ``u = Derivative(q(t), t)``).

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (LinearDamper, LinearPathway,
    ...     Point, ReferenceFrame)
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> damping = symbols('c')
    >>> pA, pB = Point('pA'), Point('pB')
    >>> pA.set_vel(N, 0)
    >>> pB.set_pos(pA, q*N.x)
    >>> pB.pos_from(pA)
    q(t)*N.x
    >>> pB.vel(N)
    Derivative(q(t), t)*N.x
    >>> linear_pathway = LinearPathway(pA, pB)
    >>> damper = LinearDamper(damping, linear_pathway)
    >>> damper
    LinearDamper(c, LinearPathway(pA, pB))

    This damper will produce a force that is proportional to both its damping
    coefficient and the pathway's extension length. Note that this force is
    negative as SymPy's sign convention for actuators is that negative forces
    are contractile and the damping force of the damper will oppose the
    direction of length change.

    >>> damper.force
    -c*sqrt(q(t)**2)*Derivative(q(t), t)/q(t)

    Parameters
    ==========

    damping : Expr
        The damping constant.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.

    See Also
    ========

    ForceActuator: force-producing actuator (superclass of ``LinearDamper``).
    LinearPathway: straight-line pathway between a pair of points.

    """

    def __init__(self, damping, pathway):
        if False:
            return 10
        'Initializer for ``LinearDamper``.\n\n        Parameters\n        ==========\n\n        damping : Expr\n            The damping constant.\n        pathway : PathwayBase\n            The pathway that the actuator follows. This must be an instance of\n            a concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.\n\n        '
        self.damping = damping
        self.pathway = pathway

    @property
    def force(self):
        if False:
            i = 10
            return i + 15
        'The damping force produced by the linear damper.'
        return -self.damping * self.pathway.extension_velocity

    @force.setter
    def force(self, force):
        if False:
            i = 10
            return i + 15
        raise AttributeError("Can't set computed attribute `force`.")

    @property
    def damping(self):
        if False:
            while True:
                i = 10
        'The damping constant for the linear damper.'
        return self._damping

    @damping.setter
    def damping(self, damping):
        if False:
            return 10
        if hasattr(self, '_damping'):
            msg = f"Can't set attribute `damping` to {repr(damping)} as it is immutable."
            raise AttributeError(msg)
        self._damping = sympify(damping, strict=True)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Representation of a ``LinearDamper``.'
        return f'{self.__class__.__name__}({self.damping}, {self.pathway})'

class TorqueActuator(ActuatorBase):
    """Torque-producing actuator.

    Explanation
    ===========

    A ``TorqueActuator`` is an actuator that produces a pair of equal and
    opposite torques on a pair of bodies.

    Examples
    ========

    To construct a torque actuator, an expression (or symbol) must be supplied
    to represent the torque it can produce, alongside a vector specifying the
    axis about which the torque will act, and a pair of frames on which the
    torque will act.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (ReferenceFrame, RigidBody,
    ...     TorqueActuator)
    >>> N = ReferenceFrame('N')
    >>> A = ReferenceFrame('A')
    >>> torque = symbols('T')
    >>> axis = N.z
    >>> parent = RigidBody('parent', frame=N)
    >>> child = RigidBody('child', frame=A)
    >>> bodies = (child, parent)
    >>> actuator = TorqueActuator(torque, axis, *bodies)
    >>> actuator
    TorqueActuator(T, axis=N.z, target_frame=A, reaction_frame=N)

    Note that because torques actually act on frames, not bodies,
    ``TorqueActuator`` will extract the frame associated with a ``RigidBody``
    when one is passed instead of a ``ReferenceFrame``.

    Parameters
    ==========

    torque : Expr
        The scalar expression defining the torque that the actuator produces.
    axis : Vector
        The axis about which the actuator applies torques.
    target_frame : ReferenceFrame | RigidBody
        The primary frame on which the actuator will apply the torque.
    reaction_frame : ReferenceFrame | RigidBody | None
        The secondary frame on which the actuator will apply the torque. Note
        that the (equal and opposite) reaction torque is applied to this frame.

    """

    def __init__(self, torque, axis, target_frame, reaction_frame=None):
        if False:
            i = 10
            return i + 15
        'Initializer for ``TorqueActuator``.\n\n        Parameters\n        ==========\n\n        torque : Expr\n            The scalar expression defining the torque that the actuator\n            produces.\n        axis : Vector\n            The axis about which the actuator applies torques.\n        target_frame : ReferenceFrame | RigidBody\n            The primary frame on which the actuator will apply the torque.\n        reaction_frame : ReferenceFrame | RigidBody | None\n           The secondary frame on which the actuator will apply the torque.\n           Note that the (equal and opposite) reaction torque is applied to\n           this frame.\n\n        '
        self.torque = torque
        self.axis = axis
        self.target_frame = target_frame
        self.reaction_frame = reaction_frame

    @classmethod
    def at_pin_joint(cls, torque, pin_joint):
        if False:
            while True:
                i = 10
        "Alternate construtor to instantiate from a ``PinJoint`` instance.\n\n        Examples\n        ========\n\n        To create a pin joint the ``PinJoint`` class requires a name, parent\n        body, and child body to be passed to its constructor. It is also\n        possible to control the joint axis using the ``joint_axis`` keyword\n        argument. In this example let's use the parent body's reference frame's\n        z-axis as the joint axis.\n\n        >>> from sympy.physics.mechanics import (PinJoint, ReferenceFrame,\n        ...     RigidBody, TorqueActuator)\n        >>> N = ReferenceFrame('N')\n        >>> A = ReferenceFrame('A')\n        >>> parent = RigidBody('parent', frame=N)\n        >>> child = RigidBody('child', frame=A)\n        >>> pin_joint = PinJoint(\n        ...     'pin',\n        ...     parent,\n        ...     child,\n        ...     joint_axis=N.z,\n        ... )\n\n        Let's also create a symbol ``T`` that will represent the torque applied\n        by the torque actuator.\n\n        >>> from sympy import symbols\n        >>> torque = symbols('T')\n\n        To create the torque actuator from the ``torque`` and ``pin_joint``\n        variables previously instantiated, these can be passed to the alternate\n        constructor class method ``at_pin_joint`` of the ``TorqueActuator``\n        class. It should be noted that a positive torque will cause a positive\n        displacement of the joint coordinate or that the torque is applied on\n        the child body with a reaction torque on the parent.\n\n        >>> actuator = TorqueActuator.at_pin_joint(torque, pin_joint)\n        >>> actuator\n        TorqueActuator(T, axis=N.z, target_frame=A, reaction_frame=N)\n\n        Parameters\n        ==========\n\n        torque : Expr\n            The scalar expression defining the torque that the actuator\n            produces.\n        pin_joint : PinJoint\n            The pin joint, and by association the parent and child bodies, on\n            which the torque actuator will act. The pair of bodies acted upon\n            by the torque actuator are the parent and child bodies of the pin\n            joint, with the child acting as the reaction body. The pin joint's\n            axis is used as the axis about which the torque actuator will apply\n            its torque.\n\n        "
        if not isinstance(pin_joint, PinJoint):
            msg = f'Value {repr(pin_joint)} passed to `pin_joint` was of type {type(pin_joint)}, must be {PinJoint}.'
            raise TypeError(msg)
        return cls(torque, pin_joint.joint_axis, pin_joint.child_interframe, pin_joint.parent_interframe)

    @property
    def torque(self):
        if False:
            for i in range(10):
                print('nop')
        'The magnitude of the torque produced by the actuator.'
        return self._torque

    @torque.setter
    def torque(self, torque):
        if False:
            print('Hello World!')
        if hasattr(self, '_torque'):
            msg = f"Can't set attribute `torque` to {repr(torque)} as it is immutable."
            raise AttributeError(msg)
        self._torque = sympify(torque, strict=True)

    @property
    def axis(self):
        if False:
            for i in range(10):
                print('nop')
        'The axis about which the torque acts.'
        return self._axis

    @axis.setter
    def axis(self, axis):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, '_axis'):
            msg = f"Can't set attribute `axis` to {repr(axis)} as it is immutable."
            raise AttributeError(msg)
        if not isinstance(axis, Vector):
            msg = f'Value {repr(axis)} passed to `axis` was of type {type(axis)}, must be {Vector}.'
            raise TypeError(msg)
        self._axis = axis

    @property
    def target_frame(self):
        if False:
            return 10
        'The primary reference frames on which the torque will act.'
        return self._target_frame

    @target_frame.setter
    def target_frame(self, target_frame):
        if False:
            return 10
        if hasattr(self, '_target_frame'):
            msg = f"Can't set attribute `target_frame` to {repr(target_frame)} as it is immutable."
            raise AttributeError(msg)
        if isinstance(target_frame, RigidBody):
            target_frame = target_frame.frame
        elif not isinstance(target_frame, ReferenceFrame):
            msg = f'Value {repr(target_frame)} passed to `target_frame` was of type {type(target_frame)}, must be {ReferenceFrame}.'
            raise TypeError(msg)
        self._target_frame = target_frame

    @property
    def reaction_frame(self):
        if False:
            return 10
        'The primary reference frames on which the torque will act.'
        return self._reaction_frame

    @reaction_frame.setter
    def reaction_frame(self, reaction_frame):
        if False:
            i = 10
            return i + 15
        if hasattr(self, '_reaction_frame'):
            msg = f"Can't set attribute `reaction_frame` to {repr(reaction_frame)} as it is immutable."
            raise AttributeError(msg)
        if isinstance(reaction_frame, RigidBody):
            reaction_frame = reaction_frame.frame
        elif not isinstance(reaction_frame, ReferenceFrame) and reaction_frame is not None:
            msg = f'Value {repr(reaction_frame)} passed to `reaction_frame` was of type {type(reaction_frame)}, must be {ReferenceFrame}.'
            raise TypeError(msg)
        self._reaction_frame = reaction_frame

    def to_loads(self):
        if False:
            for i in range(10):
                print('nop')
        "Loads required by the equations of motion method classes.\n\n        Explanation\n        ===========\n\n        ``KanesMethod`` requires a list of ``Point``-``Vector`` tuples to be\n        passed to the ``loads`` parameters of its ``kanes_equations`` method\n        when constructing the equations of motion. This method acts as a\n        utility to produce the correctly-structred pairs of points and vectors\n        required so that these can be easily concatenated with other items in\n        the list of loads and passed to ``KanesMethod.kanes_equations``. These\n        loads are also in the correct form to also be passed to the other\n        equations of motion method classes, e.g. ``LagrangesMethod``.\n\n        Examples\n        ========\n\n        The below example shows how to generate the loads produced by a torque\n        actuator that acts on a pair of bodies attached by a pin joint.\n\n        >>> from sympy import symbols\n        >>> from sympy.physics.mechanics import (PinJoint, ReferenceFrame,\n        ...     RigidBody, TorqueActuator)\n        >>> torque = symbols('T')\n        >>> N = ReferenceFrame('N')\n        >>> A = ReferenceFrame('A')\n        >>> parent = RigidBody('parent', frame=N)\n        >>> child = RigidBody('child', frame=A)\n        >>> pin_joint = PinJoint(\n        ...     'pin',\n        ...     parent,\n        ...     child,\n        ...     joint_axis=N.z,\n        ... )\n        >>> actuator = TorqueActuator.at_pin_joint(torque, pin_joint)\n\n        The forces produces by the damper can be generated by calling the\n        ``to_loads`` method.\n\n        >>> actuator.to_loads()\n        [(A, T*N.z), (N, - T*N.z)]\n\n        Alternatively, if a torque actuator is created without a reaction frame\n        then the loads returned by the ``to_loads`` method will contain just\n        the single load acting on the target frame.\n\n        >>> actuator = TorqueActuator(torque, N.z, N)\n        >>> actuator.to_loads()\n        [(N, T*N.z)]\n\n        "
        loads = [Torque(self.target_frame, self.torque * self.axis)]
        if self.reaction_frame is not None:
            loads.append(Torque(self.reaction_frame, -self.torque * self.axis))
        return loads

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Representation of a ``TorqueActuator``.'
        string = f'{self.__class__.__name__}({self.torque}, axis={self.axis}, target_frame={self.target_frame}'
        if self.reaction_frame is not None:
            string += f', reaction_frame={self.reaction_frame})'
        else:
            string += ')'
        return string