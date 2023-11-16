from sympy import Symbol
from sympy.physics.vector import Point, Vector, ReferenceFrame, Dyadic
from sympy.physics.mechanics import RigidBody, Particle, Inertia
from sympy.physics.mechanics.body_base import BodyBase
__all__ = ['Body']

class Body(RigidBody, Particle):
    """
    Body is a common representation of either a RigidBody or a Particle SymPy
    object depending on what is passed in during initialization. If a mass is
    passed in and central_inertia is left as None, the Particle object is
    created. Otherwise a RigidBody object will be created.

    Explanation
    ===========

    The attributes that Body possesses will be the same as a Particle instance
    or a Rigid Body instance depending on which was created. Additional
    attributes are listed below.

    Attributes
    ==========

    name : string
        The body's name
    masscenter : Point
        The point which represents the center of mass of the rigid body
    frame : ReferenceFrame
        The reference frame which the body is fixed in
    mass : Sympifyable
        The body's mass
    inertia : (Dyadic, Point)
        The body's inertia around its center of mass. This attribute is specific
        to the rigid body form of Body and is left undefined for the Particle
        form
    loads : iterable
        This list contains information on the different loads acting on the
        Body. Forces are listed as a (point, vector) tuple and torques are
        listed as (reference frame, vector) tuples.

    Parameters
    ==========

    name : String
        Defines the name of the body. It is used as the base for defining
        body specific properties.
    masscenter : Point, optional
        A point that represents the center of mass of the body or particle.
        If no point is given, a point is generated.
    mass : Sympifyable, optional
        A Sympifyable object which represents the mass of the body. If no
        mass is passed, one is generated.
    frame : ReferenceFrame, optional
        The ReferenceFrame that represents the reference frame of the body.
        If no frame is given, a frame is generated.
    central_inertia : Dyadic, optional
        Central inertia dyadic of the body. If none is passed while creating
        RigidBody, a default inertia is generated.

    Examples
    ========

    Default behaviour. This results in the creation of a RigidBody object for
    which the mass, mass center, frame and inertia attributes are given default
    values. ::

        >>> from sympy.physics.mechanics import Body
        >>> body = Body('name_of_body')

    This next example demonstrates the code required to specify all of the
    values of the Body object. Note this will also create a RigidBody version of
    the Body object. ::

        >>> from sympy import Symbol
        >>> from sympy.physics.mechanics import ReferenceFrame, Point, inertia
        >>> from sympy.physics.mechanics import Body
        >>> mass = Symbol('mass')
        >>> masscenter = Point('masscenter')
        >>> frame = ReferenceFrame('frame')
        >>> ixx = Symbol('ixx')
        >>> body_inertia = inertia(frame, ixx, 0, 0)
        >>> body = Body('name_of_body', masscenter, mass, frame, body_inertia)

    The minimal code required to create a Particle version of the Body object
    involves simply passing in a name and a mass. ::

        >>> from sympy import Symbol
        >>> from sympy.physics.mechanics import Body
        >>> mass = Symbol('mass')
        >>> body = Body('name_of_body', mass=mass)

    The Particle version of the Body object can also receive a masscenter point
    and a reference frame, just not an inertia.
    """

    def __init__(self, name, masscenter=None, mass=None, frame=None, central_inertia=None):
        if False:
            return 10
        self._loads = []
        if frame is None:
            frame = ReferenceFrame(name + '_frame')
        if masscenter is None:
            masscenter = Point(name + '_masscenter')
        if central_inertia is None and mass is None:
            ixx = Symbol(name + '_ixx')
            iyy = Symbol(name + '_iyy')
            izz = Symbol(name + '_izz')
            izx = Symbol(name + '_izx')
            ixy = Symbol(name + '_ixy')
            iyz = Symbol(name + '_iyz')
            _inertia = Inertia.from_inertia_scalars(masscenter, frame, ixx, iyy, izz, ixy, iyz, izx)
        else:
            _inertia = (central_inertia, masscenter)
        if mass is None:
            _mass = Symbol(name + '_mass')
        else:
            _mass = mass
        masscenter.set_vel(frame, 0)
        if central_inertia is None and mass is not None:
            BodyBase.__init__(self, name, masscenter, _mass)
            self.frame = frame
            self._central_inertia = Dyadic(0)
        else:
            BodyBase.__init__(self, name, masscenter, _mass)
            self.frame = frame
            self.inertia = _inertia

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.is_rigidbody:
            return RigidBody.__repr__(self)
        return Particle.__repr__(self)

    @property
    def loads(self):
        if False:
            while True:
                i = 10
        return self._loads

    @property
    def x(self):
        if False:
            for i in range(10):
                print('nop')
        'The basis Vector for the Body, in the x direction.'
        return self.frame.x

    @property
    def y(self):
        if False:
            i = 10
            return i + 15
        'The basis Vector for the Body, in the y direction.'
        return self.frame.y

    @property
    def z(self):
        if False:
            print('Hello World!')
        'The basis Vector for the Body, in the z direction.'
        return self.frame.z

    @property
    def inertia(self):
        if False:
            return 10
        "The body's inertia about a point; stored as (Dyadic, Point)."
        if self.is_rigidbody:
            return RigidBody.inertia.fget(self)
        return (self.central_inertia, self.masscenter)

    @inertia.setter
    def inertia(self, I):
        if False:
            return 10
        RigidBody.inertia.fset(self, I)

    @property
    def is_rigidbody(self):
        if False:
            i = 10
            return i + 15
        if hasattr(self, '_inertia'):
            return True
        return False

    def kinetic_energy(self, frame):
        if False:
            for i in range(10):
                print('nop')
        "Kinetic energy of the body.\n\n        Parameters\n        ==========\n\n        frame : ReferenceFrame or Body\n            The Body's angular velocity and the velocity of it's mass\n            center are typically defined with respect to an inertial frame but\n            any relevant frame in which the velocities are known can be supplied.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.mechanics import Body, ReferenceFrame, Point\n        >>> from sympy import symbols\n        >>> m, v, r, omega = symbols('m v r omega')\n        >>> N = ReferenceFrame('N')\n        >>> O = Point('O')\n        >>> P = Body('P', masscenter=O, mass=m)\n        >>> P.masscenter.set_vel(N, v * N.y)\n        >>> P.kinetic_energy(N)\n        m*v**2/2\n\n        >>> N = ReferenceFrame('N')\n        >>> b = ReferenceFrame('b')\n        >>> b.set_ang_vel(N, omega * b.x)\n        >>> P = Point('P')\n        >>> P.set_vel(N, v * N.x)\n        >>> B = Body('B', masscenter=P, frame=b)\n        >>> B.kinetic_energy(N)\n        B_ixx*omega**2/2 + B_mass*v**2/2\n\n        See Also\n        ========\n\n        sympy.physics.mechanics : Particle, RigidBody\n\n        "
        if isinstance(frame, Body):
            frame = Body.frame
        if self.is_rigidbody:
            return RigidBody(self.name, self.masscenter, self.frame, self.mass, (self.central_inertia, self.masscenter)).kinetic_energy(frame)
        return Particle(self.name, self.masscenter, self.mass).kinetic_energy(frame)

    def apply_force(self, force, point=None, reaction_body=None, reaction_point=None):
        if False:
            while True:
                i = 10
        "Add force to the body(s).\n\n        Explanation\n        ===========\n\n        Applies the force on self or equal and oppposite forces on\n        self and other body if both are given on the desried point on the bodies.\n        The force applied on other body is taken opposite of self, i.e, -force.\n\n        Parameters\n        ==========\n\n        force: Vector\n            The force to be applied.\n        point: Point, optional\n            The point on self on which force is applied.\n            By default self's masscenter.\n        reaction_body: Body, optional\n            Second body on which equal and opposite force\n            is to be applied.\n        reaction_point : Point, optional\n            The point on other body on which equal and opposite\n            force is applied. By default masscenter of other body.\n\n        Example\n        =======\n\n        >>> from sympy import symbols\n        >>> from sympy.physics.mechanics import Body, Point, dynamicsymbols\n        >>> m, g = symbols('m g')\n        >>> B = Body('B')\n        >>> force1 = m*g*B.z\n        >>> B.apply_force(force1) #Applying force on B's masscenter\n        >>> B.loads\n        [(B_masscenter, g*m*B_frame.z)]\n\n        We can also remove some part of force from any point on the body by\n        adding the opposite force to the body on that point.\n\n        >>> f1, f2 = dynamicsymbols('f1 f2')\n        >>> P = Point('P') #Considering point P on body B\n        >>> B.apply_force(f1*B.x + f2*B.y, P)\n        >>> B.loads\n        [(B_masscenter, g*m*B_frame.z), (P, f1(t)*B_frame.x + f2(t)*B_frame.y)]\n\n        Let's remove f1 from point P on body B.\n\n        >>> B.apply_force(-f1*B.x, P)\n        >>> B.loads\n        [(B_masscenter, g*m*B_frame.z), (P, f2(t)*B_frame.y)]\n\n        To further demonstrate the use of ``apply_force`` attribute,\n        consider two bodies connected through a spring.\n\n        >>> from sympy.physics.mechanics import Body, dynamicsymbols\n        >>> N = Body('N') #Newtonion Frame\n        >>> x = dynamicsymbols('x')\n        >>> B1 = Body('B1')\n        >>> B2 = Body('B2')\n        >>> spring_force = x*N.x\n\n        Now let's apply equal and opposite spring force to the bodies.\n\n        >>> P1 = Point('P1')\n        >>> P2 = Point('P2')\n        >>> B1.apply_force(spring_force, point=P1, reaction_body=B2, reaction_point=P2)\n\n        We can check the loads(forces) applied to bodies now.\n\n        >>> B1.loads\n        [(P1, x(t)*N_frame.x)]\n        >>> B2.loads\n        [(P2, - x(t)*N_frame.x)]\n\n        Notes\n        =====\n\n        If a new force is applied to a body on a point which already has some\n        force applied on it, then the new force is added to the already applied\n        force on that point.\n\n        "
        if not isinstance(point, Point):
            if point is None:
                point = self.masscenter
            else:
                raise TypeError('Force must be applied to a point on the body.')
        if not isinstance(force, Vector):
            raise TypeError('Force must be a vector.')
        if reaction_body is not None:
            reaction_body.apply_force(-force, point=reaction_point)
        for load in self._loads:
            if point in load:
                force += load[1]
                self._loads.remove(load)
                break
        self._loads.append((point, force))

    def apply_torque(self, torque, reaction_body=None):
        if False:
            i = 10
            return i + 15
        "Add torque to the body(s).\n\n        Explanation\n        ===========\n\n        Applies the torque on self or equal and oppposite torquess on\n        self and other body if both are given.\n        The torque applied on other body is taken opposite of self,\n        i.e, -torque.\n\n        Parameters\n        ==========\n\n        torque: Vector\n            The torque to be applied.\n        reaction_body: Body, optional\n            Second body on which equal and opposite torque\n            is to be applied.\n\n        Example\n        =======\n\n        >>> from sympy import symbols\n        >>> from sympy.physics.mechanics import Body, dynamicsymbols\n        >>> t = symbols('t')\n        >>> B = Body('B')\n        >>> torque1 = t*B.z\n        >>> B.apply_torque(torque1)\n        >>> B.loads\n        [(B_frame, t*B_frame.z)]\n\n        We can also remove some part of torque from the body by\n        adding the opposite torque to the body.\n\n        >>> t1, t2 = dynamicsymbols('t1 t2')\n        >>> B.apply_torque(t1*B.x + t2*B.y)\n        >>> B.loads\n        [(B_frame, t1(t)*B_frame.x + t2(t)*B_frame.y + t*B_frame.z)]\n\n        Let's remove t1 from Body B.\n\n        >>> B.apply_torque(-t1*B.x)\n        >>> B.loads\n        [(B_frame, t2(t)*B_frame.y + t*B_frame.z)]\n\n        To further demonstrate the use, let us consider two bodies such that\n        a torque `T` is acting on one body, and `-T` on the other.\n\n        >>> from sympy.physics.mechanics import Body, dynamicsymbols\n        >>> N = Body('N') #Newtonion frame\n        >>> B1 = Body('B1')\n        >>> B2 = Body('B2')\n        >>> v = dynamicsymbols('v')\n        >>> T = v*N.y #Torque\n\n        Now let's apply equal and opposite torque to the bodies.\n\n        >>> B1.apply_torque(T, B2)\n\n        We can check the loads (torques) applied to bodies now.\n\n        >>> B1.loads\n        [(B1_frame, v(t)*N_frame.y)]\n        >>> B2.loads\n        [(B2_frame, - v(t)*N_frame.y)]\n\n        Notes\n        =====\n\n        If a new torque is applied on body which already has some torque applied on it,\n        then the new torque is added to the previous torque about the body's frame.\n\n        "
        if not isinstance(torque, Vector):
            raise TypeError('A Vector must be supplied to add torque.')
        if reaction_body is not None:
            reaction_body.apply_torque(-torque)
        for load in self._loads:
            if self.frame in load:
                torque += load[1]
                self._loads.remove(load)
                break
        self._loads.append((self.frame, torque))

    def clear_loads(self):
        if False:
            return 10
        "\n        Clears the Body's loads list.\n\n        Example\n        =======\n\n        >>> from sympy.physics.mechanics import Body\n        >>> B = Body('B')\n        >>> force = B.x + B.y\n        >>> B.apply_force(force)\n        >>> B.loads\n        [(B_masscenter, B_frame.x + B_frame.y)]\n        >>> B.clear_loads()\n        >>> B.loads\n        []\n\n        "
        self._loads = []

    def remove_load(self, about=None):
        if False:
            print('Hello World!')
        "\n        Remove load about a point or frame.\n\n        Parameters\n        ==========\n\n        about : Point or ReferenceFrame, optional\n            The point about which force is applied,\n            and is to be removed.\n            If about is None, then the torque about\n            self's frame is removed.\n\n        Example\n        =======\n\n        >>> from sympy.physics.mechanics import Body, Point\n        >>> B = Body('B')\n        >>> P = Point('P')\n        >>> f1 = B.x\n        >>> f2 = B.y\n        >>> B.apply_force(f1)\n        >>> B.apply_force(f2, P)\n        >>> B.loads\n        [(B_masscenter, B_frame.x), (P, B_frame.y)]\n\n        >>> B.remove_load(P)\n        >>> B.loads\n        [(B_masscenter, B_frame.x)]\n\n        "
        if about is not None:
            if not isinstance(about, Point):
                raise TypeError('Load is applied about Point or ReferenceFrame.')
        else:
            about = self.frame
        for load in self._loads:
            if about in load:
                self._loads.remove(load)
                break

    def masscenter_vel(self, body):
        if False:
            i = 10
            return i + 15
        "\n        Returns the velocity of the mass center with respect to the provided\n        rigid body or reference frame.\n\n        Parameters\n        ==========\n\n        body: Body or ReferenceFrame\n            The rigid body or reference frame to calculate the velocity in.\n\n        Example\n        =======\n\n        >>> from sympy.physics.mechanics import Body\n        >>> A = Body('A')\n        >>> B = Body('B')\n        >>> A.masscenter.set_vel(B.frame, 5*B.frame.x)\n        >>> A.masscenter_vel(B)\n        5*B_frame.x\n        >>> A.masscenter_vel(B.frame)\n        5*B_frame.x\n\n        "
        if isinstance(body, ReferenceFrame):
            frame = body
        elif isinstance(body, Body):
            frame = body.frame
        return self.masscenter.vel(frame)

    def ang_vel_in(self, body):
        if False:
            return 10
        "\n        Returns this body's angular velocity with respect to the provided\n        rigid body or reference frame.\n\n        Parameters\n        ==========\n\n        body: Body or ReferenceFrame\n            The rigid body or reference frame to calculate the angular velocity in.\n\n        Example\n        =======\n\n        >>> from sympy.physics.mechanics import Body, ReferenceFrame\n        >>> A = Body('A')\n        >>> N = ReferenceFrame('N')\n        >>> B = Body('B', frame=N)\n        >>> A.frame.set_ang_vel(N, 5*N.x)\n        >>> A.ang_vel_in(B)\n        5*N.x\n        >>> A.ang_vel_in(N)\n        5*N.x\n\n        "
        if isinstance(body, ReferenceFrame):
            frame = body
        elif isinstance(body, Body):
            frame = body.frame
        return self.frame.ang_vel_in(frame)

    def dcm(self, body):
        if False:
            i = 10
            return i + 15
        "\n        Returns the direction cosine matrix of this body relative to the\n        provided rigid body or reference frame.\n\n        Parameters\n        ==========\n\n        body: Body or ReferenceFrame\n            The rigid body or reference frame to calculate the dcm.\n\n        Example\n        =======\n\n        >>> from sympy.physics.mechanics import Body\n        >>> A = Body('A')\n        >>> B = Body('B')\n        >>> A.frame.orient_axis(B.frame, B.frame.x, 5)\n        >>> A.dcm(B)\n        Matrix([\n        [1,       0,      0],\n        [0,  cos(5), sin(5)],\n        [0, -sin(5), cos(5)]])\n        >>> A.dcm(B.frame)\n        Matrix([\n        [1,       0,      0],\n        [0,  cos(5), sin(5)],\n        [0, -sin(5), cos(5)]])\n\n        "
        if isinstance(body, ReferenceFrame):
            frame = body
        elif isinstance(body, Body):
            frame = body.frame
        return self.frame.dcm(frame)

    def parallel_axis(self, point, frame=None):
        if False:
            return 10
        "Returns the inertia dyadic of the body with respect to another\n        point.\n\n        Parameters\n        ==========\n\n        point : sympy.physics.vector.Point\n            The point to express the inertia dyadic about.\n        frame : sympy.physics.vector.ReferenceFrame\n            The reference frame used to construct the dyadic.\n\n        Returns\n        =======\n\n        inertia : sympy.physics.vector.Dyadic\n            The inertia dyadic of the rigid body expressed about the provided\n            point.\n\n        Example\n        =======\n\n        >>> from sympy.physics.mechanics import Body\n        >>> A = Body('A')\n        >>> P = A.masscenter.locatenew('point', 3 * A.x + 5 * A.y)\n        >>> A.parallel_axis(P).to_matrix(A.frame)\n        Matrix([\n        [A_ixx + 25*A_mass, A_ixy - 15*A_mass,             A_izx],\n        [A_ixy - 15*A_mass,  A_iyy + 9*A_mass,             A_iyz],\n        [            A_izx,             A_iyz, A_izz + 34*A_mass]])\n\n        "
        if self.is_rigidbody:
            return RigidBody.parallel_axis(self, point, frame)
        return Particle.parallel_axis(self, point, frame)