from sympy.core import S, Basic, Dict, Symbol, Tuple, sympify
from sympy.core.symbol import Str
from sympy.sets import Set, FiniteSet, EmptySet
from sympy.utilities.iterables import iterable

class Class(Set):
    """
    The base class for any kind of class in the set-theoretic sense.

    Explanation
    ===========

    In axiomatic set theories, everything is a class.  A class which
    can be a member of another class is a set.  A class which is not a
    member of another class is a proper class.  The class `\\{1, 2\\}`
    is a set; the class of all sets is a proper class.

    This class is essentially a synonym for :class:`sympy.core.Set`.
    The goal of this class is to assure easier migration to the
    eventual proper implementation of set theory.
    """
    is_proper = False

class Object(Symbol):
    """
    The base class for any kind of object in an abstract category.

    Explanation
    ===========

    While technically any instance of :class:`~.Basic` will do, this
    class is the recommended way to create abstract objects in
    abstract categories.
    """

class Morphism(Basic):
    """
    The base class for any morphism in an abstract category.

    Explanation
    ===========

    In abstract categories, a morphism is an arrow between two
    category objects.  The object where the arrow starts is called the
    domain, while the object where the arrow ends is called the
    codomain.

    Two morphisms between the same pair of objects are considered to
    be the same morphisms.  To distinguish between morphisms between
    the same objects use :class:`NamedMorphism`.

    It is prohibited to instantiate this class.  Use one of the
    derived classes instead.

    See Also
    ========

    IdentityMorphism, NamedMorphism, CompositeMorphism
    """

    def __new__(cls, domain, codomain):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Cannot instantiate Morphism.  Use derived classes instead.')

    @property
    def domain(self):
        if False:
            print('Hello World!')
        '\n        Returns the domain of the morphism.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> f.domain\n        Object("A")\n\n        '
        return self.args[0]

    @property
    def codomain(self):
        if False:
            return 10
        '\n        Returns the codomain of the morphism.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> f.codomain\n        Object("B")\n\n        '
        return self.args[1]

    def compose(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Composes self with the supplied morphism.\n\n        The order of elements in the composition is the usual order,\n        i.e., to construct `g\\circ f` use ``g.compose(f)``.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> g * f\n        CompositeMorphism((NamedMorphism(Object("A"), Object("B"), "f"),\n        NamedMorphism(Object("B"), Object("C"), "g")))\n        >>> (g * f).domain\n        Object("A")\n        >>> (g * f).codomain\n        Object("C")\n\n        '
        return CompositeMorphism(other, self)

    def __mul__(self, other):
        if False:
            return 10
        '\n        Composes self with the supplied morphism.\n\n        The semantics of this operation is given by the following\n        equation: ``g * f == g.compose(f)`` for composable morphisms\n        ``g`` and ``f``.\n\n        See Also\n        ========\n\n        compose\n        '
        return self.compose(other)

class IdentityMorphism(Morphism):
    """
    Represents an identity morphism.

    Explanation
    ===========

    An identity morphism is a morphism with equal domain and codomain,
    which acts as an identity with respect to composition.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, IdentityMorphism
    >>> A = Object("A")
    >>> B = Object("B")
    >>> f = NamedMorphism(A, B, "f")
    >>> id_A = IdentityMorphism(A)
    >>> id_B = IdentityMorphism(B)
    >>> f * id_A == f
    True
    >>> id_B * f == f
    True

    See Also
    ========

    Morphism
    """

    def __new__(cls, domain):
        if False:
            i = 10
            return i + 15
        return Basic.__new__(cls, domain)

    @property
    def codomain(self):
        if False:
            while True:
                i = 10
        return self.domain

class NamedMorphism(Morphism):
    """
    Represents a morphism which has a name.

    Explanation
    ===========

    Names are used to distinguish between morphisms which have the
    same domain and codomain: two named morphisms are equal if they
    have the same domains, codomains, and names.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism
    >>> A = Object("A")
    >>> B = Object("B")
    >>> f = NamedMorphism(A, B, "f")
    >>> f
    NamedMorphism(Object("A"), Object("B"), "f")
    >>> f.name
    'f'

    See Also
    ========

    Morphism
    """

    def __new__(cls, domain, codomain, name):
        if False:
            while True:
                i = 10
        if not name:
            raise ValueError('Empty morphism names not allowed.')
        if not isinstance(name, Str):
            name = Str(name)
        return Basic.__new__(cls, domain, codomain, name)

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the name of the morphism.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> f.name\n        \'f\'\n\n        '
        return self.args[2].name

class CompositeMorphism(Morphism):
    """
    Represents a morphism which is a composition of other morphisms.

    Explanation
    ===========

    Two composite morphisms are equal if the morphisms they were
    obtained from (components) are the same and were listed in the
    same order.

    The arguments to the constructor for this class should be listed
    in diagram order: to obtain the composition `g\\circ f` from the
    instances of :class:`Morphism` ``g`` and ``f`` use
    ``CompositeMorphism(f, g)``.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, CompositeMorphism
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> g * f
    CompositeMorphism((NamedMorphism(Object("A"), Object("B"), "f"),
    NamedMorphism(Object("B"), Object("C"), "g")))
    >>> CompositeMorphism(f, g) == g * f
    True

    """

    @staticmethod
    def _add_morphism(t, morphism):
        if False:
            for i in range(10):
                print('nop')
        '\n        Intelligently adds ``morphism`` to tuple ``t``.\n\n        Explanation\n        ===========\n\n        If ``morphism`` is a composite morphism, its components are\n        added to the tuple.  If ``morphism`` is an identity, nothing\n        is added to the tuple.\n\n        No composability checks are performed.\n        '
        if isinstance(morphism, CompositeMorphism):
            return t + morphism.components
        elif isinstance(morphism, IdentityMorphism):
            return t
        else:
            return t + Tuple(morphism)

    def __new__(cls, *components):
        if False:
            while True:
                i = 10
        if components and (not isinstance(components[0], Morphism)):
            return CompositeMorphism.__new__(cls, *components[0])
        normalised_components = Tuple()
        for (current, following) in zip(components, components[1:]):
            if not isinstance(current, Morphism) or not isinstance(following, Morphism):
                raise TypeError('All components must be morphisms.')
            if current.codomain != following.domain:
                raise ValueError('Uncomposable morphisms.')
            normalised_components = CompositeMorphism._add_morphism(normalised_components, current)
        normalised_components = CompositeMorphism._add_morphism(normalised_components, components[-1])
        if not normalised_components:
            return components[0]
        elif len(normalised_components) == 1:
            return normalised_components[0]
        return Basic.__new__(cls, normalised_components)

    @property
    def components(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the components of this composite morphism.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> (g * f).components\n        (NamedMorphism(Object("A"), Object("B"), "f"),\n        NamedMorphism(Object("B"), Object("C"), "g"))\n\n        '
        return self.args[0]

    @property
    def domain(self):
        if False:
            while True:
                i = 10
        '\n        Returns the domain of this composite morphism.\n\n        The domain of the composite morphism is the domain of its\n        first component.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> (g * f).domain\n        Object("A")\n\n        '
        return self.components[0].domain

    @property
    def codomain(self):
        if False:
            while True:
                i = 10
        '\n        Returns the codomain of this composite morphism.\n\n        The codomain of the composite morphism is the codomain of its\n        last component.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> (g * f).codomain\n        Object("C")\n\n        '
        return self.components[-1].codomain

    def flatten(self, new_name):
        if False:
            return 10
        '\n        Forgets the composite structure of this morphism.\n\n        Explanation\n        ===========\n\n        If ``new_name`` is not empty, returns a :class:`NamedMorphism`\n        with the supplied name, otherwise returns a :class:`Morphism`.\n        In both cases the domain of the new morphism is the domain of\n        this composite morphism and the codomain of the new morphism\n        is the codomain of this composite morphism.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> (g * f).flatten("h")\n        NamedMorphism(Object("A"), Object("C"), "h")\n\n        '
        return NamedMorphism(self.domain, self.codomain, new_name)

class Category(Basic):
    """
    An (abstract) category.

    Explanation
    ===========

    A category [JoyOfCats] is a quadruple `\\mbox{K} = (O, \\hom, id,
    \\circ)` consisting of

    * a (set-theoretical) class `O`, whose members are called
      `K`-objects,

    * for each pair `(A, B)` of `K`-objects, a set `\\hom(A, B)` whose
      members are called `K`-morphisms from `A` to `B`,

    * for a each `K`-object `A`, a morphism `id:A\\rightarrow A`,
      called the `K`-identity of `A`,

    * a composition law `\\circ` associating with every `K`-morphisms
      `f:A\\rightarrow B` and `g:B\\rightarrow C` a `K`-morphism `g\\circ
      f:A\\rightarrow C`, called the composite of `f` and `g`.

    Composition is associative, `K`-identities are identities with
    respect to composition, and the sets `\\hom(A, B)` are pairwise
    disjoint.

    This class knows nothing about its objects and morphisms.
    Concrete cases of (abstract) categories should be implemented as
    classes derived from this one.

    Certain instances of :class:`Diagram` can be asserted to be
    commutative in a :class:`Category` by supplying the argument
    ``commutative_diagrams`` in the constructor.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, Diagram, Category
    >>> from sympy import FiniteSet
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> d = Diagram([f, g])
    >>> K = Category("K", commutative_diagrams=[d])
    >>> K.commutative_diagrams == FiniteSet(d)
    True

    See Also
    ========

    Diagram
    """

    def __new__(cls, name, objects=EmptySet, commutative_diagrams=EmptySet):
        if False:
            print('Hello World!')
        if not name:
            raise ValueError('A Category cannot have an empty name.')
        if not isinstance(name, Str):
            name = Str(name)
        if not isinstance(objects, Class):
            objects = Class(objects)
        new_category = Basic.__new__(cls, name, objects, FiniteSet(*commutative_diagrams))
        return new_category

    @property
    def name(self):
        if False:
            print('Hello World!')
        '\n        Returns the name of this category.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Category\n        >>> K = Category("K")\n        >>> K.name\n        \'K\'\n\n        '
        return self.args[0].name

    @property
    def objects(self):
        if False:
            return 10
        '\n        Returns the class of objects of this category.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, Category\n        >>> from sympy import FiniteSet\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> K = Category("K", FiniteSet(A, B))\n        >>> K.objects\n        Class({Object("A"), Object("B")})\n\n        '
        return self.args[1]

    @property
    def commutative_diagrams(self):
        if False:
            return 10
        '\n        Returns the :class:`~.FiniteSet` of diagrams which are known to\n        be commutative in this category.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism, Diagram, Category\n        >>> from sympy import FiniteSet\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> d = Diagram([f, g])\n        >>> K = Category("K", commutative_diagrams=[d])\n        >>> K.commutative_diagrams == FiniteSet(d)\n        True\n\n        '
        return self.args[2]

    def hom(self, A, B):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('hom-sets are not implemented in Category.')

    def all_morphisms(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Obtaining the class of morphisms is not implemented in Category.')

class Diagram(Basic):
    """
    Represents a diagram in a certain category.

    Explanation
    ===========

    Informally, a diagram is a collection of objects of a category and
    certain morphisms between them.  A diagram is still a monoid with
    respect to morphism composition; i.e., identity morphisms, as well
    as all composites of morphisms included in the diagram belong to
    the diagram.  For a more formal approach to this notion see
    [Pare1970].

    The components of composite morphisms are also added to the
    diagram.  No properties are assigned to such morphisms by default.

    A commutative diagram is often accompanied by a statement of the
    following kind: "if such morphisms with such properties exist,
    then such morphisms which such properties exist and the diagram is
    commutative".  To represent this, an instance of :class:`Diagram`
    includes a collection of morphisms which are the premises and
    another collection of conclusions.  ``premises`` and
    ``conclusions`` associate morphisms belonging to the corresponding
    categories with the :class:`~.FiniteSet`'s of their properties.

    The set of properties of a composite morphism is the intersection
    of the sets of properties of its components.  The domain and
    codomain of a conclusion morphism should be among the domains and
    codomains of the morphisms listed as the premises of a diagram.

    No checks are carried out of whether the supplied object and
    morphisms do belong to one and the same category.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, Diagram
    >>> from sympy import pprint, default_sort_key
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> d = Diagram([f, g])
    >>> premises_keys = sorted(d.premises.keys(), key=default_sort_key)
    >>> pprint(premises_keys, use_unicode=False)
    [g*f:A-->C, id:A-->A, id:B-->B, id:C-->C, f:A-->B, g:B-->C]
    >>> pprint(d.premises, use_unicode=False)
    {g*f:A-->C: EmptySet, id:A-->A: EmptySet, id:B-->B: EmptySet, id:C-->C: EmptyS >
    <BLANKLINE>
    > et, f:A-->B: EmptySet, g:B-->C: EmptySet}
    >>> d = Diagram([f, g], {g * f: "unique"})
    >>> pprint(d.conclusions,use_unicode=False)
    {g*f:A-->C: {unique}}

    References
    ==========

    [Pare1970] B. Pareigis: Categories and functors.  Academic Press, 1970.

    """

    @staticmethod
    def _set_dict_union(dictionary, key, value):
        if False:
            i = 10
            return i + 15
        '\n        If ``key`` is in ``dictionary``, set the new value of ``key``\n        to be the union between the old value and ``value``.\n        Otherwise, set the value of ``key`` to ``value.\n\n        Returns ``True`` if the key already was in the dictionary and\n        ``False`` otherwise.\n        '
        if key in dictionary:
            dictionary[key] = dictionary[key] | value
            return True
        else:
            dictionary[key] = value
            return False

    @staticmethod
    def _add_morphism_closure(morphisms, morphism, props, add_identities=True, recurse_composites=True):
        if False:
            print('Hello World!')
        '\n        Adds a morphism and its attributes to the supplied dictionary\n        ``morphisms``.  If ``add_identities`` is True, also adds the\n        identity morphisms for the domain and the codomain of\n        ``morphism``.\n        '
        if not Diagram._set_dict_union(morphisms, morphism, props):
            if isinstance(morphism, IdentityMorphism):
                if props:
                    raise ValueError('Instances of IdentityMorphism cannot have properties.')
                return
            if add_identities:
                empty = EmptySet
                id_dom = IdentityMorphism(morphism.domain)
                id_cod = IdentityMorphism(morphism.codomain)
                Diagram._set_dict_union(morphisms, id_dom, empty)
                Diagram._set_dict_union(morphisms, id_cod, empty)
            for (existing_morphism, existing_props) in list(morphisms.items()):
                new_props = existing_props & props
                if morphism.domain == existing_morphism.codomain:
                    left = morphism * existing_morphism
                    Diagram._set_dict_union(morphisms, left, new_props)
                if morphism.codomain == existing_morphism.domain:
                    right = existing_morphism * morphism
                    Diagram._set_dict_union(morphisms, right, new_props)
            if isinstance(morphism, CompositeMorphism) and recurse_composites:
                empty = EmptySet
                for component in morphism.components:
                    Diagram._add_morphism_closure(morphisms, component, empty, add_identities)

    def __new__(cls, *args):
        if False:
            print('Hello World!')
        '\n        Construct a new instance of Diagram.\n\n        Explanation\n        ===========\n\n        If no arguments are supplied, an empty diagram is created.\n\n        If at least an argument is supplied, ``args[0]`` is\n        interpreted as the premises of the diagram.  If ``args[0]`` is\n        a list, it is interpreted as a list of :class:`Morphism`\'s, in\n        which each :class:`Morphism` has an empty set of properties.\n        If ``args[0]`` is a Python dictionary or a :class:`Dict`, it\n        is interpreted as a dictionary associating to some\n        :class:`Morphism`\'s some properties.\n\n        If at least two arguments are supplied ``args[1]`` is\n        interpreted as the conclusions of the diagram.  The type of\n        ``args[1]`` is interpreted in exactly the same way as the type\n        of ``args[0]``.  If only one argument is supplied, the diagram\n        has no conclusions.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism\n        >>> from sympy.categories import IdentityMorphism, Diagram\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> d = Diagram([f, g])\n        >>> IdentityMorphism(A) in d.premises.keys()\n        True\n        >>> g * f in d.premises.keys()\n        True\n        >>> d = Diagram([f, g], {g * f: "unique"})\n        >>> d.conclusions[g * f]\n        {unique}\n\n        '
        premises = {}
        conclusions = {}
        objects = EmptySet
        if len(args) >= 1:
            premises_arg = args[0]
            if isinstance(premises_arg, list):
                empty = EmptySet
                for morphism in premises_arg:
                    objects |= FiniteSet(morphism.domain, morphism.codomain)
                    Diagram._add_morphism_closure(premises, morphism, empty)
            elif isinstance(premises_arg, (dict, Dict)):
                for (morphism, props) in premises_arg.items():
                    objects |= FiniteSet(morphism.domain, morphism.codomain)
                    Diagram._add_morphism_closure(premises, morphism, FiniteSet(*props) if iterable(props) else FiniteSet(props))
        if len(args) >= 2:
            conclusions_arg = args[1]
            if isinstance(conclusions_arg, list):
                empty = EmptySet
                for morphism in conclusions_arg:
                    if sympify(objects.contains(morphism.domain)) is S.true and sympify(objects.contains(morphism.codomain)) is S.true:
                        Diagram._add_morphism_closure(conclusions, morphism, empty, add_identities=False, recurse_composites=False)
            elif isinstance(conclusions_arg, (dict, Dict)):
                for (morphism, props) in conclusions_arg.items():
                    if morphism.domain in objects and morphism.codomain in objects:
                        Diagram._add_morphism_closure(conclusions, morphism, FiniteSet(*props) if iterable(props) else FiniteSet(props), add_identities=False, recurse_composites=False)
        return Basic.__new__(cls, Dict(premises), Dict(conclusions), objects)

    @property
    def premises(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the premises of this diagram.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism\n        >>> from sympy.categories import IdentityMorphism, Diagram\n        >>> from sympy import pretty\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> id_A = IdentityMorphism(A)\n        >>> id_B = IdentityMorphism(B)\n        >>> d = Diagram([f])\n        >>> print(pretty(d.premises, use_unicode=False))\n        {id:A-->A: EmptySet, id:B-->B: EmptySet, f:A-->B: EmptySet}\n\n        '
        return self.args[0]

    @property
    def conclusions(self):
        if False:
            print('Hello World!')
        '\n        Returns the conclusions of this diagram.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism\n        >>> from sympy.categories import IdentityMorphism, Diagram\n        >>> from sympy import FiniteSet\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> d = Diagram([f, g])\n        >>> IdentityMorphism(A) in d.premises.keys()\n        True\n        >>> g * f in d.premises.keys()\n        True\n        >>> d = Diagram([f, g], {g * f: "unique"})\n        >>> d.conclusions[g * f] == FiniteSet("unique")\n        True\n\n        '
        return self.args[1]

    @property
    def objects(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the :class:`~.FiniteSet` of objects that appear in this\n        diagram.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism, Diagram\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> d = Diagram([f, g])\n        >>> d.objects\n        {Object("A"), Object("B"), Object("C")}\n\n        '
        return self.args[2]

    def hom(self, A, B):
        if False:
            i = 10
            return i + 15
        '\n        Returns a 2-tuple of sets of morphisms between objects ``A`` and\n        ``B``: one set of morphisms listed as premises, and the other set\n        of morphisms listed as conclusions.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism, Diagram\n        >>> from sympy import pretty\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> d = Diagram([f, g], {g * f: "unique"})\n        >>> print(pretty(d.hom(A, C), use_unicode=False))\n        ({g*f:A-->C}, {g*f:A-->C})\n\n        See Also\n        ========\n        Object, Morphism\n        '
        premises = EmptySet
        conclusions = EmptySet
        for morphism in self.premises.keys():
            if morphism.domain == A and morphism.codomain == B:
                premises |= FiniteSet(morphism)
        for morphism in self.conclusions.keys():
            if morphism.domain == A and morphism.codomain == B:
                conclusions |= FiniteSet(morphism)
        return (premises, conclusions)

    def is_subdiagram(self, diagram):
        if False:
            print('Hello World!')
        '\n        Checks whether ``diagram`` is a subdiagram of ``self``.\n        Diagram `D\'` is a subdiagram of `D` if all premises\n        (conclusions) of `D\'` are contained in the premises\n        (conclusions) of `D`.  The morphisms contained\n        both in `D\'` and `D` should have the same properties for `D\'`\n        to be a subdiagram of `D`.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism, Diagram\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> d = Diagram([f, g], {g * f: "unique"})\n        >>> d1 = Diagram([f])\n        >>> d.is_subdiagram(d1)\n        True\n        >>> d1.is_subdiagram(d)\n        False\n        '
        premises = all((m in self.premises and diagram.premises[m] == self.premises[m] for m in diagram.premises))
        if not premises:
            return False
        conclusions = all((m in self.conclusions and diagram.conclusions[m] == self.conclusions[m] for m in diagram.conclusions))
        return conclusions

    def subdiagram_from_objects(self, objects):
        if False:
            return 10
        '\n        If ``objects`` is a subset of the objects of ``self``, returns\n        a diagram which has as premises all those premises of ``self``\n        which have a domains and codomains in ``objects``, likewise\n        for conclusions.  Properties are preserved.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism, Diagram\n        >>> from sympy import FiniteSet\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> d = Diagram([f, g], {f: "unique", g*f: "veryunique"})\n        >>> d1 = d.subdiagram_from_objects(FiniteSet(A, B))\n        >>> d1 == Diagram([f], {f: "unique"})\n        True\n        '
        if not objects.is_subset(self.objects):
            raise ValueError('Supplied objects should all belong to the diagram.')
        new_premises = {}
        for (morphism, props) in self.premises.items():
            if sympify(objects.contains(morphism.domain)) is S.true and sympify(objects.contains(morphism.codomain)) is S.true:
                new_premises[morphism] = props
        new_conclusions = {}
        for (morphism, props) in self.conclusions.items():
            if sympify(objects.contains(morphism.domain)) is S.true and sympify(objects.contains(morphism.codomain)) is S.true:
                new_conclusions[morphism] = props
        return Diagram(new_premises, new_conclusions)