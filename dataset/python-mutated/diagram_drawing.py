"""
This module contains the functionality to arrange the nodes of a
diagram on an abstract grid, and then to produce a graphical
representation of the grid.

The currently supported back-ends are Xy-pic [Xypic].

Layout Algorithm
================

This section provides an overview of the algorithms implemented in
:class:`DiagramGrid` to lay out diagrams.

The first step of the algorithm is the removal composite and identity
morphisms which do not have properties in the supplied diagram.  The
premises and conclusions of the diagram are then merged.

The generic layout algorithm begins with the construction of the
"skeleton" of the diagram.  The skeleton is an undirected graph which
has the objects of the diagram as vertices and has an (undirected)
edge between each pair of objects between which there exist morphisms.
The direction of the morphisms does not matter at this stage.  The
skeleton also includes an edge between each pair of vertices `A` and
`C` such that there exists an object `B` which is connected via
a morphism to `A`, and via a morphism to `C`.

The skeleton constructed in this way has the property that every
object is a vertex of a triangle formed by three edges of the
skeleton.  This property lies at the base of the generic layout
algorithm.

After the skeleton has been constructed, the algorithm lists all
triangles which can be formed.  Note that some triangles will not have
all edges corresponding to morphisms which will actually be drawn.
Triangles which have only one edge or less which will actually be
drawn are immediately discarded.

The list of triangles is sorted according to the number of edges which
correspond to morphisms, then the triangle with the least number of such
edges is selected.  One of such edges is picked and the corresponding
objects are placed horizontally, on a grid.  This edge is recorded to
be in the fringe.  The algorithm then finds a "welding" of a triangle
to the fringe.  A welding is an edge in the fringe where a triangle
could be attached.  If the algorithm succeeds in finding such a
welding, it adds to the grid that vertex of the triangle which was not
yet included in any edge in the fringe and records the two new edges in
the fringe.  This process continues iteratively until all objects of
the diagram has been placed or until no more weldings can be found.

An edge is only removed from the fringe when a welding to this edge
has been found, and there is no room around this edge to place
another vertex.

When no more weldings can be found, but there are still triangles
left, the algorithm searches for a possibility of attaching one of the
remaining triangles to the existing structure by a vertex.  If such a
possibility is found, the corresponding edge of the found triangle is
placed in the found space and the iterative process of welding
triangles restarts.

When logical groups are supplied, each of these groups is laid out
independently.  Then a diagram is constructed in which groups are
objects and any two logical groups between which there exist morphisms
are connected via a morphism.  This diagram is laid out.  Finally,
the grid which includes all objects of the initial diagram is
constructed by replacing the cells which contain logical groups with
the corresponding laid out grids, and by correspondingly expanding the
rows and columns.

The sequential layout algorithm begins by constructing the
underlying undirected graph defined by the morphisms obtained after
simplifying premises and conclusions and merging them (see above).
The vertex with the minimal degree is then picked up and depth-first
search is started from it.  All objects which are located at distance
`n` from the root in the depth-first search tree, are positioned in
the `n`-th column of the resulting grid.  The sequential layout will
therefore attempt to lay the objects out along a line.

References
==========

.. [Xypic] https://xy-pic.sourceforge.net/

"""
from sympy.categories import CompositeMorphism, IdentityMorphism, NamedMorphism, Diagram
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
__doctest_requires__ = {('preview_diagram',): 'pyglet'}

class _GrowableGrid:
    """
    Holds a growable grid of objects.

    Explanation
    ===========

    It is possible to append or prepend a row or a column to the grid
    using the corresponding methods.  Prepending rows or columns has
    the effect of changing the coordinates of the already existing
    elements.

    This class currently represents a naive implementation of the
    functionality with little attempt at optimisation.
    """

    def __init__(self, width, height):
        if False:
            return 10
        self._width = width
        self._height = height
        self._array = [[None for j in range(width)] for i in range(height)]

    @property
    def width(self):
        if False:
            print('Hello World!')
        return self._width

    @property
    def height(self):
        if False:
            i = 10
            return i + 15
        return self._height

    def __getitem__(self, i_j):
        if False:
            return 10
        '\n        Returns the element located at in the i-th line and j-th\n        column.\n        '
        (i, j) = i_j
        return self._array[i][j]

    def __setitem__(self, i_j, newvalue):
        if False:
            return 10
        '\n        Sets the element located at in the i-th line and j-th\n        column.\n        '
        (i, j) = i_j
        self._array[i][j] = newvalue

    def append_row(self):
        if False:
            while True:
                i = 10
        '\n        Appends an empty row to the grid.\n        '
        self._height += 1
        self._array.append([None for j in range(self._width)])

    def append_column(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Appends an empty column to the grid.\n        '
        self._width += 1
        for i in range(self._height):
            self._array[i].append(None)

    def prepend_row(self):
        if False:
            while True:
                i = 10
        '\n        Prepends the grid with an empty row.\n        '
        self._height += 1
        self._array.insert(0, [None for j in range(self._width)])

    def prepend_column(self):
        if False:
            i = 10
            return i + 15
        '\n        Prepends the grid with an empty column.\n        '
        self._width += 1
        for i in range(self._height):
            self._array[i].insert(0, None)

class DiagramGrid:
    """
    Constructs and holds the fitting of the diagram into a grid.

    Explanation
    ===========

    The mission of this class is to analyse the structure of the
    supplied diagram and to place its objects on a grid such that,
    when the objects and the morphisms are actually drawn, the diagram
    would be "readable", in the sense that there will not be many
    intersections of moprhisms.  This class does not perform any
    actual drawing.  It does strive nevertheless to offer sufficient
    metadata to draw a diagram.

    Consider the following simple diagram.

    >>> from sympy.categories import Object, NamedMorphism
    >>> from sympy.categories import Diagram, DiagramGrid
    >>> from sympy import pprint
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> diagram = Diagram([f, g])

    The simplest way to have a diagram laid out is the following:

    >>> grid = DiagramGrid(diagram)
    >>> (grid.width, grid.height)
    (2, 2)
    >>> pprint(grid)
    A  B
    <BLANKLINE>
       C

    Sometimes one sees the diagram as consisting of logical groups.
    One can advise ``DiagramGrid`` as to such groups by employing the
    ``groups`` keyword argument.

    Consider the following diagram:

    >>> D = Object("D")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> h = NamedMorphism(D, A, "h")
    >>> k = NamedMorphism(D, B, "k")
    >>> diagram = Diagram([f, g, h, k])

    Lay it out with generic layout:

    >>> grid = DiagramGrid(diagram)
    >>> pprint(grid)
    A  B  D
    <BLANKLINE>
       C

    Now, we can group the objects `A` and `D` to have them near one
    another:

    >>> grid = DiagramGrid(diagram, groups=[[A, D], B, C])
    >>> pprint(grid)
    B     C
    <BLANKLINE>
    A  D

    Note how the positioning of the other objects changes.

    Further indications can be supplied to the constructor of
    :class:`DiagramGrid` using keyword arguments.  The currently
    supported hints are explained in the following paragraphs.

    :class:`DiagramGrid` does not automatically guess which layout
    would suit the supplied diagram better.  Consider, for example,
    the following linear diagram:

    >>> E = Object("E")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> h = NamedMorphism(C, D, "h")
    >>> i = NamedMorphism(D, E, "i")
    >>> diagram = Diagram([f, g, h, i])

    When laid out with the generic layout, it does not get to look
    linear:

    >>> grid = DiagramGrid(diagram)
    >>> pprint(grid)
    A  B
    <BLANKLINE>
       C  D
    <BLANKLINE>
          E

    To get it laid out in a line, use ``layout="sequential"``:

    >>> grid = DiagramGrid(diagram, layout="sequential")
    >>> pprint(grid)
    A  B  C  D  E

    One may sometimes need to transpose the resulting layout.  While
    this can always be done by hand, :class:`DiagramGrid` provides a
    hint for that purpose:

    >>> grid = DiagramGrid(diagram, layout="sequential", transpose=True)
    >>> pprint(grid)
    A
    <BLANKLINE>
    B
    <BLANKLINE>
    C
    <BLANKLINE>
    D
    <BLANKLINE>
    E

    Separate hints can also be provided for each group.  For an
    example, refer to ``tests/test_drawing.py``, and see the different
    ways in which the five lemma [FiveLemma] can be laid out.

    See Also
    ========

    Diagram

    References
    ==========

    .. [FiveLemma] https://en.wikipedia.org/wiki/Five_lemma
    """

    @staticmethod
    def _simplify_morphisms(morphisms):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given a dictionary mapping morphisms to their properties,\n        returns a new dictionary in which there are no morphisms which\n        do not have properties, and which are compositions of other\n        morphisms included in the dictionary.  Identities are dropped\n        as well.\n        '
        newmorphisms = {}
        for (morphism, props) in morphisms.items():
            if isinstance(morphism, CompositeMorphism) and (not props):
                continue
            elif isinstance(morphism, IdentityMorphism):
                continue
            else:
                newmorphisms[morphism] = props
        return newmorphisms

    @staticmethod
    def _merge_premises_conclusions(premises, conclusions):
        if False:
            return 10
        '\n        Given two dictionaries of morphisms and their properties,\n        produces a single dictionary which includes elements from both\n        dictionaries.  If a morphism has some properties in premises\n        and also in conclusions, the properties in conclusions take\n        priority.\n        '
        return dict(chain(premises.items(), conclusions.items()))

    @staticmethod
    def _juxtapose_edges(edge1, edge2):
        if False:
            print('Hello World!')
        '\n        If ``edge1`` and ``edge2`` have precisely one common endpoint,\n        returns an edge which would form a triangle with ``edge1`` and\n        ``edge2``.\n\n        If ``edge1`` and ``edge2`` do not have a common endpoint,\n        returns ``None``.\n\n        If ``edge1`` and ``edge`` are the same edge, returns ``None``.\n        '
        intersection = edge1 & edge2
        if len(intersection) != 1:
            return None
        return edge1 - intersection | edge2 - intersection

    @staticmethod
    def _add_edge_append(dictionary, edge, elem):
        if False:
            i = 10
            return i + 15
        '\n        If ``edge`` is not in ``dictionary``, adds ``edge`` to the\n        dictionary and sets its value to ``[elem]``.  Otherwise\n        appends ``elem`` to the value of existing entry.\n\n        Note that edges are undirected, thus `(A, B) = (B, A)`.\n        '
        if edge in dictionary:
            dictionary[edge].append(elem)
        else:
            dictionary[edge] = [elem]

    @staticmethod
    def _build_skeleton(morphisms):
        if False:
            i = 10
            return i + 15
        '\n        Creates a dictionary which maps edges to corresponding\n        morphisms.  Thus for a morphism `f:A\rightarrow B`, the edge\n        `(A, B)` will be associated with `f`.  This function also adds\n        to the list those edges which are formed by juxtaposition of\n        two edges already in the list.  These new edges are not\n        associated with any morphism and are only added to assure that\n        the diagram can be decomposed into triangles.\n        '
        edges = {}
        for morphism in morphisms:
            DiagramGrid._add_edge_append(edges, frozenset([morphism.domain, morphism.codomain]), morphism)
        edges1 = dict(edges)
        for w in edges1:
            for v in edges1:
                wv = DiagramGrid._juxtapose_edges(w, v)
                if wv and wv not in edges:
                    edges[wv] = []
        return edges

    @staticmethod
    def _list_triangles(edges):
        if False:
            return 10
        '\n        Builds the set of triangles formed by the supplied edges.  The\n        triangles are arbitrary and need not be commutative.  A\n        triangle is a set that contains all three of its sides.\n        '
        triangles = set()
        for w in edges:
            for v in edges:
                wv = DiagramGrid._juxtapose_edges(w, v)
                if wv and wv in edges:
                    triangles.add(frozenset([w, v, wv]))
        return triangles

    @staticmethod
    def _drop_redundant_triangles(triangles, skeleton):
        if False:
            print('Hello World!')
        '\n        Returns a list which contains only those triangles who have\n        morphisms associated with at least two edges.\n        '
        return [tri for tri in triangles if len([e for e in tri if skeleton[e]]) >= 2]

    @staticmethod
    def _morphism_length(morphism):
        if False:
            i = 10
            return i + 15
        '\n        Returns the length of a morphism.  The length of a morphism is\n        the number of components it consists of.  A non-composite\n        morphism is of length 1.\n        '
        if isinstance(morphism, CompositeMorphism):
            return len(morphism.components)
        else:
            return 1

    @staticmethod
    def _compute_triangle_min_sizes(triangles, edges):
        if False:
            while True:
                i = 10
        '\n        Returns a dictionary mapping triangles to their minimal sizes.\n        The minimal size of a triangle is the sum of maximal lengths\n        of morphisms associated to the sides of the triangle.  The\n        length of a morphism is the number of components it consists\n        of.  A non-composite morphism is of length 1.\n\n        Sorting triangles by this metric attempts to address two\n        aspects of layout.  For triangles with only simple morphisms\n        in the edge, this assures that triangles with all three edges\n        visible will get typeset after triangles with less visible\n        edges, which sometimes minimizes the necessity in diagonal\n        arrows.  For triangles with composite morphisms in the edges,\n        this assures that objects connected with shorter morphisms\n        will be laid out first, resulting the visual proximity of\n        those objects which are connected by shorter morphisms.\n        '
        triangle_sizes = {}
        for triangle in triangles:
            size = 0
            for e in triangle:
                morphisms = edges[e]
                if morphisms:
                    size += max((DiagramGrid._morphism_length(m) for m in morphisms))
            triangle_sizes[triangle] = size
        return triangle_sizes

    @staticmethod
    def _triangle_objects(triangle):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given a triangle, returns the objects included in it.\n        '
        return frozenset(chain(*tuple(triangle)))

    @staticmethod
    def _other_vertex(triangle, edge):
        if False:
            i = 10
            return i + 15
        '\n        Given a triangle and an edge of it, returns the vertex which\n        opposes the edge.\n        '
        return list(DiagramGrid._triangle_objects(triangle) - set(edge))[0]

    @staticmethod
    def _empty_point(pt, grid):
        if False:
            return 10
        '\n        Checks if the cell at coordinates ``pt`` is either empty or\n        out of the bounds of the grid.\n        '
        if pt[0] < 0 or pt[1] < 0 or pt[0] >= grid.height or (pt[1] >= grid.width):
            return True
        return grid[pt] is None

    @staticmethod
    def _put_object(coords, obj, grid, fringe):
        if False:
            while True:
                i = 10
        '\n        Places an object at the coordinate ``cords`` in ``grid``,\n        growing the grid and updating ``fringe``, if necessary.\n        Returns (0, 0) if no row or column has been prepended, (1, 0)\n        if a row was prepended, (0, 1) if a column was prepended and\n        (1, 1) if both a column and a row were prepended.\n        '
        (i, j) = coords
        offset = (0, 0)
        if i == -1:
            grid.prepend_row()
            i = 0
            offset = (1, 0)
            for k in range(len(fringe)):
                ((i1, j1), (i2, j2)) = fringe[k]
                fringe[k] = ((i1 + 1, j1), (i2 + 1, j2))
        elif i == grid.height:
            grid.append_row()
        if j == -1:
            j = 0
            offset = (offset[0], 1)
            grid.prepend_column()
            for k in range(len(fringe)):
                ((i1, j1), (i2, j2)) = fringe[k]
                fringe[k] = ((i1, j1 + 1), (i2, j2 + 1))
        elif j == grid.width:
            grid.append_column()
        grid[i, j] = obj
        return offset

    @staticmethod
    def _choose_target_cell(pt1, pt2, edge, obj, skeleton, grid):
        if False:
            print('Hello World!')
        '\n        Given two points, ``pt1`` and ``pt2``, and the welding edge\n        ``edge``, chooses one of the two points to place the opposing\n        vertex ``obj`` of the triangle.  If neither of this points\n        fits, returns ``None``.\n        '
        pt1_empty = DiagramGrid._empty_point(pt1, grid)
        pt2_empty = DiagramGrid._empty_point(pt2, grid)
        if pt1_empty and pt2_empty:
            A = grid[edge[0]]
            if skeleton.get(frozenset([A, obj])):
                return pt1
            else:
                return pt2
        if pt1_empty:
            return pt1
        elif pt2_empty:
            return pt2
        else:
            return None

    @staticmethod
    def _find_triangle_to_weld(triangles, fringe, grid):
        if False:
            for i in range(10):
                print('nop')
        '\n        Finds, if possible, a triangle and an edge in the ``fringe`` to\n        which the triangle could be attached.  Returns the tuple\n        containing the triangle and the index of the corresponding\n        edge in the ``fringe``.\n\n        This function relies on the fact that objects are unique in\n        the diagram.\n        '
        for triangle in triangles:
            for (a, b) in fringe:
                if frozenset([grid[a], grid[b]]) in triangle:
                    return (triangle, (a, b))
        return None

    @staticmethod
    def _weld_triangle(tri, welding_edge, fringe, grid, skeleton):
        if False:
            while True:
                i = 10
        '\n        If possible, welds the triangle ``tri`` to ``fringe`` and\n        returns ``False``.  If this method encounters a degenerate\n        situation in the fringe and corrects it such that a restart of\n        the search is required, it returns ``True`` (which means that\n        a restart in finding triangle weldings is required).\n\n        A degenerate situation is a situation when an edge listed in\n        the fringe does not belong to the visual boundary of the\n        diagram.\n        '
        (a, b) = welding_edge
        target_cell = None
        obj = DiagramGrid._other_vertex(tri, (grid[a], grid[b]))
        if abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1:
            target_cell = (a[0], b[1])
            if grid[target_cell]:
                target_cell = (b[0], a[1])
                if grid[target_cell]:
                    fringe.remove((a, b))
                    return True
        elif a[0] == b[0]:
            down_left = (a[0] + 1, a[1])
            down_right = (a[0] + 1, b[1])
            target_cell = DiagramGrid._choose_target_cell(down_left, down_right, (a, b), obj, skeleton, grid)
            if not target_cell:
                up_left = (a[0] - 1, a[1])
                up_right = (a[0] - 1, b[1])
                target_cell = DiagramGrid._choose_target_cell(up_left, up_right, (a, b), obj, skeleton, grid)
                if not target_cell:
                    fringe.remove((a, b))
                    return True
        elif a[1] == b[1]:
            right_up = (a[0], a[1] + 1)
            right_down = (b[0], a[1] + 1)
            target_cell = DiagramGrid._choose_target_cell(right_up, right_down, (a, b), obj, skeleton, grid)
            if not target_cell:
                left_up = (a[0], a[1] - 1)
                left_down = (b[0], a[1] - 1)
                target_cell = DiagramGrid._choose_target_cell(left_up, left_down, (a, b), obj, skeleton, grid)
                if not target_cell:
                    fringe.remove((a, b))
                    return True
        offset = DiagramGrid._put_object(target_cell, obj, grid, fringe)
        target_cell = (target_cell[0] + offset[0], target_cell[1] + offset[1])
        a = (a[0] + offset[0], a[1] + offset[1])
        b = (b[0] + offset[0], b[1] + offset[1])
        fringe.extend([(a, target_cell), (b, target_cell)])
        return False

    @staticmethod
    def _triangle_key(tri, triangle_sizes):
        if False:
            print('Hello World!')
        '\n        Returns a key for the supplied triangle.  It should be the\n        same independently of the hash randomisation.\n        '
        objects = sorted(DiagramGrid._triangle_objects(tri), key=default_sort_key)
        return (triangle_sizes[tri], default_sort_key(objects))

    @staticmethod
    def _pick_root_edge(tri, skeleton):
        if False:
            while True:
                i = 10
        '\n        For a given triangle always picks the same root edge.  The\n        root edge is the edge that will be placed first on the grid.\n        '
        candidates = [sorted(e, key=default_sort_key) for e in tri if skeleton[e]]
        sorted_candidates = sorted(candidates, key=default_sort_key)
        return tuple(sorted(sorted_candidates[0], key=default_sort_key))

    @staticmethod
    def _drop_irrelevant_triangles(triangles, placed_objects):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns only those triangles whose set of objects is not\n        completely included in ``placed_objects``.\n        '
        return [tri for tri in triangles if not placed_objects.issuperset(DiagramGrid._triangle_objects(tri))]

    @staticmethod
    def _grow_pseudopod(triangles, fringe, grid, skeleton, placed_objects):
        if False:
            for i in range(10):
                print('nop')
        '\n        Starting from an object in the existing structure on the ``grid``,\n        adds an edge to which a triangle from ``triangles`` could be\n        welded.  If this method has found a way to do so, it returns\n        the object it has just added.\n\n        This method should be applied when ``_weld_triangle`` cannot\n        find weldings any more.\n        '
        for i in range(grid.height):
            for j in range(grid.width):
                obj = grid[i, j]
                if not obj:
                    continue

                def good_triangle(tri):
                    if False:
                        for i in range(10):
                            print('nop')
                    objs = DiagramGrid._triangle_objects(tri)
                    return obj in objs and placed_objects & objs - {obj} == set()
                tris = [tri for tri in triangles if good_triangle(tri)]
                if not tris:
                    continue
                tri = tris[0]
                candidates = sorted([e for e in tri if skeleton[e]], key=lambda e: FiniteSet(*e).sort_key())
                edges = [e for e in candidates if obj in e]
                edge = edges[0]
                other_obj = tuple(edge - frozenset([obj]))[0]
                neighbours = [(i - 1, j), (i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)]
                for pt in neighbours:
                    if DiagramGrid._empty_point(pt, grid):
                        offset = DiagramGrid._put_object(pt, other_obj, grid, fringe)
                        i += offset[0]
                        j += offset[1]
                        pt = (pt[0] + offset[0], pt[1] + offset[1])
                        fringe.append(((i, j), pt))
                        return other_obj
        return None

    @staticmethod
    def _handle_groups(diagram, groups, merged_morphisms, hints):
        if False:
            while True:
                i = 10
        '\n        Given the slightly preprocessed morphisms of the diagram,\n        produces a grid laid out according to ``groups``.\n\n        If a group has hints, it is laid out with those hints only,\n        without any influence from ``hints``.  Otherwise, it is laid\n        out with ``hints``.\n        '

        def lay_out_group(group, local_hints):
            if False:
                while True:
                    i = 10
            '\n            If ``group`` is a set of objects, uses a ``DiagramGrid``\n            to lay it out and returns the grid.  Otherwise returns the\n            object (i.e., ``group``).  If ``local_hints`` is not\n            empty, it is supplied to ``DiagramGrid`` as the dictionary\n            of hints.  Otherwise, the ``hints`` argument of\n            ``_handle_groups`` is used.\n            '
            if isinstance(group, FiniteSet):
                for obj in group:
                    obj_groups[obj] = group
                if local_hints:
                    groups_grids[group] = DiagramGrid(diagram.subdiagram_from_objects(group), **local_hints)
                else:
                    groups_grids[group] = DiagramGrid(diagram.subdiagram_from_objects(group), **hints)
            else:
                obj_groups[group] = group

        def group_to_finiteset(group):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Converts ``group`` to a :class:``FiniteSet`` if it is an\n            iterable.\n            '
            if iterable(group):
                return FiniteSet(*group)
            else:
                return group
        obj_groups = {}
        groups_grids = {}
        if isinstance(groups, (dict, Dict)):
            finiteset_groups = {}
            for (group, local_hints) in groups.items():
                finiteset_group = group_to_finiteset(group)
                finiteset_groups[finiteset_group] = local_hints
                lay_out_group(group, local_hints)
            groups = finiteset_groups
        else:
            finiteset_groups = []
            for group in groups:
                finiteset_group = group_to_finiteset(group)
                finiteset_groups.append(finiteset_group)
                lay_out_group(finiteset_group, None)
            groups = finiteset_groups
        new_morphisms = []
        for morphism in merged_morphisms:
            dom = obj_groups[morphism.domain]
            cod = obj_groups[morphism.codomain]
            if dom != cod:
                new_morphisms.append(NamedMorphism(dom, cod, 'dummy'))
        top_grid = DiagramGrid(Diagram(new_morphisms))

        def group_size(group):
            if False:
                return 10
            '\n            For the supplied group (or object, eventually), returns\n            the size of the cell that will hold this group (object).\n            '
            if group in groups_grids:
                grid = groups_grids[group]
                return (grid.height, grid.width)
            else:
                return (1, 1)
        row_heights = [max((group_size(top_grid[i, j])[0] for j in range(top_grid.width))) for i in range(top_grid.height)]
        column_widths = [max((group_size(top_grid[i, j])[1] for i in range(top_grid.height))) for j in range(top_grid.width)]
        grid = _GrowableGrid(sum(column_widths), sum(row_heights))
        real_row = 0
        real_column = 0
        for logical_row in range(top_grid.height):
            for logical_column in range(top_grid.width):
                obj = top_grid[logical_row, logical_column]
                if obj in groups_grids:
                    local_grid = groups_grids[obj]
                    for i in range(local_grid.height):
                        for j in range(local_grid.width):
                            grid[real_row + i, real_column + j] = local_grid[i, j]
                else:
                    grid[real_row, real_column] = obj
                real_column += column_widths[logical_column]
            real_column = 0
            real_row += row_heights[logical_row]
        return grid

    @staticmethod
    def _generic_layout(diagram, merged_morphisms):
        if False:
            i = 10
            return i + 15
        '\n        Produces the generic layout for the supplied diagram.\n        '
        all_objects = set(diagram.objects)
        if len(all_objects) == 1:
            grid = _GrowableGrid(1, 1)
            grid[0, 0] = tuple(all_objects)[0]
            return grid
        skeleton = DiagramGrid._build_skeleton(merged_morphisms)
        grid = _GrowableGrid(2, 1)
        if len(skeleton) == 1:
            objects = sorted(all_objects, key=default_sort_key)
            grid[0, 0] = objects[0]
            grid[0, 1] = objects[1]
            return grid
        triangles = DiagramGrid._list_triangles(skeleton)
        triangles = DiagramGrid._drop_redundant_triangles(triangles, skeleton)
        triangle_sizes = DiagramGrid._compute_triangle_min_sizes(triangles, skeleton)
        triangles = sorted(triangles, key=lambda tri: DiagramGrid._triangle_key(tri, triangle_sizes))
        root_edge = DiagramGrid._pick_root_edge(triangles[0], skeleton)
        (grid[0, 0], grid[0, 1]) = root_edge
        fringe = [((0, 0), (0, 1))]
        placed_objects = set(root_edge)
        while placed_objects != all_objects:
            welding = DiagramGrid._find_triangle_to_weld(triangles, fringe, grid)
            if welding:
                (triangle, welding_edge) = welding
                restart_required = DiagramGrid._weld_triangle(triangle, welding_edge, fringe, grid, skeleton)
                if restart_required:
                    continue
                placed_objects.update(DiagramGrid._triangle_objects(triangle))
            else:
                new_obj = DiagramGrid._grow_pseudopod(triangles, fringe, grid, skeleton, placed_objects)
                if not new_obj:
                    remaining_objects = all_objects - placed_objects
                    remaining_diagram = diagram.subdiagram_from_objects(FiniteSet(*remaining_objects))
                    remaining_grid = DiagramGrid(remaining_diagram)
                    final_width = grid.width + remaining_grid.width
                    final_height = max(grid.height, remaining_grid.height)
                    final_grid = _GrowableGrid(final_width, final_height)
                    for i in range(grid.width):
                        for j in range(grid.height):
                            final_grid[i, j] = grid[i, j]
                    start_j = grid.width
                    for i in range(remaining_grid.height):
                        for j in range(remaining_grid.width):
                            final_grid[i, start_j + j] = remaining_grid[i, j]
                    return final_grid
                placed_objects.add(new_obj)
            triangles = DiagramGrid._drop_irrelevant_triangles(triangles, placed_objects)
        return grid

    @staticmethod
    def _get_undirected_graph(objects, merged_morphisms):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given the objects and the relevant morphisms of a diagram,\n        returns the adjacency lists of the underlying undirected\n        graph.\n        '
        adjlists = {}
        for obj in objects:
            adjlists[obj] = []
        for morphism in merged_morphisms:
            adjlists[morphism.domain].append(morphism.codomain)
            adjlists[morphism.codomain].append(morphism.domain)
        for obj in adjlists.keys():
            adjlists[obj].sort(key=default_sort_key)
        return adjlists

    @staticmethod
    def _sequential_layout(diagram, merged_morphisms):
        if False:
            while True:
                i = 10
        '\n        Lays out the diagram in "sequential" layout.  This method\n        will attempt to produce a result as close to a line as\n        possible.  For linear diagrams, the result will actually be a\n        line.\n        '
        objects = diagram.objects
        sorted_objects = sorted(objects, key=default_sort_key)
        adjlists = DiagramGrid._get_undirected_graph(objects, merged_morphisms)
        root = sorted_objects[0]
        mindegree = len(adjlists[root])
        for obj in sorted_objects:
            current_degree = len(adjlists[obj])
            if current_degree < mindegree:
                root = obj
                mindegree = current_degree
        grid = _GrowableGrid(1, 1)
        grid[0, 0] = root
        placed_objects = {root}

        def place_objects(pt, placed_objects):
            if False:
                i = 10
                return i + 15
            '\n            Does depth-first search in the underlying graph of the\n            diagram and places the objects en route.\n            '
            new_pt = (pt[0], pt[1] + 1)
            for adjacent_obj in adjlists[grid[pt]]:
                if adjacent_obj in placed_objects:
                    continue
                DiagramGrid._put_object(new_pt, adjacent_obj, grid, [])
                placed_objects.add(adjacent_obj)
                placed_objects.update(place_objects(new_pt, placed_objects))
                new_pt = (new_pt[0] + 1, new_pt[1])
            return placed_objects
        place_objects((0, 0), placed_objects)
        return grid

    @staticmethod
    def _drop_inessential_morphisms(merged_morphisms):
        if False:
            for i in range(10):
                print('nop')
        '\n        Removes those morphisms which should appear in the diagram,\n        but which have no relevance to object layout.\n\n        Currently this removes "loop" morphisms: the non-identity\n        morphisms with the same domains and codomains.\n        '
        morphisms = [m for m in merged_morphisms if m.domain != m.codomain]
        return morphisms

    @staticmethod
    def _get_connected_components(objects, merged_morphisms):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given a container of morphisms, returns a list of connected\n        components formed by these morphisms.  A connected component\n        is represented by a diagram consisting of the corresponding\n        morphisms.\n        '
        component_index = {}
        for o in objects:
            component_index[o] = None
        adjlist = DiagramGrid._get_undirected_graph(objects, merged_morphisms)

        def traverse_component(object, current_index):
            if False:
                print('Hello World!')
            '\n            Does a depth-first search traversal of the component\n            containing ``object``.\n            '
            component_index[object] = current_index
            for o in adjlist[object]:
                if component_index[o] is None:
                    traverse_component(o, current_index)
        current_index = 0
        for o in adjlist:
            if component_index[o] is None:
                traverse_component(o, current_index)
                current_index += 1
        component_objects = [[] for i in range(current_index)]
        for (o, idx) in component_index.items():
            component_objects[idx].append(o)
        component_morphisms = []
        for component in component_objects:
            current_morphisms = {}
            for m in merged_morphisms:
                if m.domain in component and m.codomain in component:
                    current_morphisms[m] = merged_morphisms[m]
            if len(component) == 1:
                current_morphisms[IdentityMorphism(component[0])] = FiniteSet()
            component_morphisms.append(Diagram(current_morphisms))
        return component_morphisms

    def __init__(self, diagram, groups=None, **hints):
        if False:
            for i in range(10):
                print('nop')
        premises = DiagramGrid._simplify_morphisms(diagram.premises)
        conclusions = DiagramGrid._simplify_morphisms(diagram.conclusions)
        all_merged_morphisms = DiagramGrid._merge_premises_conclusions(premises, conclusions)
        merged_morphisms = DiagramGrid._drop_inessential_morphisms(all_merged_morphisms)
        self._morphisms = all_merged_morphisms
        components = DiagramGrid._get_connected_components(diagram.objects, all_merged_morphisms)
        if groups and groups != diagram.objects:
            self._grid = DiagramGrid._handle_groups(diagram, groups, merged_morphisms, hints)
        elif len(components) > 1:
            grids = []
            components = sorted(components, key=default_sort_key)
            for component in components:
                grid = DiagramGrid(component, **hints)
                grids.append(grid)
            total_width = sum((g.width for g in grids))
            total_height = max((g.height for g in grids))
            grid = _GrowableGrid(total_width, total_height)
            start_j = 0
            for g in grids:
                for i in range(g.height):
                    for j in range(g.width):
                        grid[i, start_j + j] = g[i, j]
                start_j += g.width
            self._grid = grid
        elif 'layout' in hints:
            if hints['layout'] == 'sequential':
                self._grid = DiagramGrid._sequential_layout(diagram, merged_morphisms)
        else:
            self._grid = DiagramGrid._generic_layout(diagram, merged_morphisms)
        if hints.get('transpose'):
            grid = _GrowableGrid(self._grid.height, self._grid.width)
            for i in range(self._grid.height):
                for j in range(self._grid.width):
                    grid[j, i] = self._grid[i, j]
            self._grid = grid

    @property
    def width(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the number of columns in this diagram layout.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism\n        >>> from sympy.categories import Diagram, DiagramGrid\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> diagram = Diagram([f, g])\n        >>> grid = DiagramGrid(diagram)\n        >>> grid.width\n        2\n\n        '
        return self._grid.width

    @property
    def height(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the number of rows in this diagram layout.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism\n        >>> from sympy.categories import Diagram, DiagramGrid\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> diagram = Diagram([f, g])\n        >>> grid = DiagramGrid(diagram)\n        >>> grid.height\n        2\n\n        '
        return self._grid.height

    def __getitem__(self, i_j):
        if False:
            return 10
        '\n        Returns the object placed in the row ``i`` and column ``j``.\n        The indices are 0-based.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism\n        >>> from sympy.categories import Diagram, DiagramGrid\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> diagram = Diagram([f, g])\n        >>> grid = DiagramGrid(diagram)\n        >>> (grid[0, 0], grid[0, 1])\n        (Object("A"), Object("B"))\n        >>> (grid[1, 0], grid[1, 1])\n        (None, Object("C"))\n\n        '
        (i, j) = i_j
        return self._grid[i, j]

    @property
    def morphisms(self):
        if False:
            print('Hello World!')
        '\n        Returns those morphisms (and their properties) which are\n        sufficiently meaningful to be drawn.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism\n        >>> from sympy.categories import Diagram, DiagramGrid\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> diagram = Diagram([f, g])\n        >>> grid = DiagramGrid(diagram)\n        >>> grid.morphisms\n        {NamedMorphism(Object("A"), Object("B"), "f"): EmptySet,\n        NamedMorphism(Object("B"), Object("C"), "g"): EmptySet}\n\n        '
        return self._morphisms

    def __str__(self):
        if False:
            i = 10
            return i + 15
        '\n        Produces a string representation of this class.\n\n        This method returns a string representation of the underlying\n        list of lists of objects.\n\n        Examples\n        ========\n\n        >>> from sympy.categories import Object, NamedMorphism\n        >>> from sympy.categories import Diagram, DiagramGrid\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> diagram = Diagram([f, g])\n        >>> grid = DiagramGrid(diagram)\n        >>> print(grid)\n        [[Object("A"), Object("B")],\n        [None, Object("C")]]\n\n        '
        return repr(self._grid._array)

class ArrowStringDescription:
    """
    Stores the information necessary for producing an Xy-pic
    description of an arrow.

    The principal goal of this class is to abstract away the string
    representation of an arrow and to also provide the functionality
    to produce the actual Xy-pic string.

    ``unit`` sets the unit which will be used to specify the amount of
    curving and other distances.  ``horizontal_direction`` should be a
    string of ``"r"`` or ``"l"`` specifying the horizontal offset of the
    target cell of the arrow relatively to the current one.
    ``vertical_direction`` should  specify the vertical offset using a
    series of either ``"d"`` or ``"u"``.  ``label_position`` should be
    either ``"^"``, ``"_"``,  or ``"|"`` to specify that the label should
    be positioned above the arrow, below the arrow or just over the arrow,
    in a break.  Note that the notions "above" and "below" are relative
    to arrow direction.  ``label`` stores the morphism label.

    This works as follows (disregard the yet unexplained arguments):

    >>> from sympy.categories.diagram_drawing import ArrowStringDescription
    >>> astr = ArrowStringDescription(
    ... unit="mm", curving=None, curving_amount=None,
    ... looping_start=None, looping_end=None, horizontal_direction="d",
    ... vertical_direction="r", label_position="_", label="f")
    >>> print(str(astr))
    \\ar[dr]_{f}

    ``curving`` should be one of ``"^"``, ``"_"`` to specify in which
    direction the arrow is going to curve. ``curving_amount`` is a number
    describing how many ``unit``'s the morphism is going to curve:

    >>> astr = ArrowStringDescription(
    ... unit="mm", curving="^", curving_amount=12,
    ... looping_start=None, looping_end=None, horizontal_direction="d",
    ... vertical_direction="r", label_position="_", label="f")
    >>> print(str(astr))
    \\ar@/^12mm/[dr]_{f}

    ``looping_start`` and ``looping_end`` are currently only used for
    loop morphisms, those which have the same domain and codomain.
    These two attributes should store a valid Xy-pic direction and
    specify, correspondingly, the direction the arrow gets out into
    and the direction the arrow gets back from:

    >>> astr = ArrowStringDescription(
    ... unit="mm", curving=None, curving_amount=None,
    ... looping_start="u", looping_end="l", horizontal_direction="",
    ... vertical_direction="", label_position="_", label="f")
    >>> print(str(astr))
    \\ar@(u,l)[]_{f}

    ``label_displacement`` controls how far the arrow label is from
    the ends of the arrow.  For example, to position the arrow label
    near the arrow head, use ">":

    >>> astr = ArrowStringDescription(
    ... unit="mm", curving="^", curving_amount=12,
    ... looping_start=None, looping_end=None, horizontal_direction="d",
    ... vertical_direction="r", label_position="_", label="f")
    >>> astr.label_displacement = ">"
    >>> print(str(astr))
    \\ar@/^12mm/[dr]_>{f}

    Finally, ``arrow_style`` is used to specify the arrow style.  To
    get a dashed arrow, for example, use "{-->}" as arrow style:

    >>> astr = ArrowStringDescription(
    ... unit="mm", curving="^", curving_amount=12,
    ... looping_start=None, looping_end=None, horizontal_direction="d",
    ... vertical_direction="r", label_position="_", label="f")
    >>> astr.arrow_style = "{-->}"
    >>> print(str(astr))
    \\ar@/^12mm/@{-->}[dr]_{f}

    Notes
    =====

    Instances of :class:`ArrowStringDescription` will be constructed
    by :class:`XypicDiagramDrawer` and provided for further use in
    formatters.  The user is not expected to construct instances of
    :class:`ArrowStringDescription` themselves.

    To be able to properly utilise this class, the reader is encouraged
    to checkout the Xy-pic user guide, available at [Xypic].

    See Also
    ========

    XypicDiagramDrawer

    References
    ==========

    .. [Xypic] https://xy-pic.sourceforge.net/
    """

    def __init__(self, unit, curving, curving_amount, looping_start, looping_end, horizontal_direction, vertical_direction, label_position, label):
        if False:
            for i in range(10):
                print('nop')
        self.unit = unit
        self.curving = curving
        self.curving_amount = curving_amount
        self.looping_start = looping_start
        self.looping_end = looping_end
        self.horizontal_direction = horizontal_direction
        self.vertical_direction = vertical_direction
        self.label_position = label_position
        self.label = label
        self.label_displacement = ''
        self.arrow_style = ''
        self.forced_label_position = False

    def __str__(self):
        if False:
            i = 10
            return i + 15
        if self.curving:
            curving_str = '@/%s%d%s/' % (self.curving, self.curving_amount, self.unit)
        else:
            curving_str = ''
        if self.looping_start and self.looping_end:
            looping_str = '@(%s,%s)' % (self.looping_start, self.looping_end)
        else:
            looping_str = ''
        if self.arrow_style:
            style_str = '@' + self.arrow_style
        else:
            style_str = ''
        return '\\ar%s%s%s[%s%s]%s%s{%s}' % (curving_str, looping_str, style_str, self.horizontal_direction, self.vertical_direction, self.label_position, self.label_displacement, self.label)

class XypicDiagramDrawer:
    """
    Given a :class:`~.Diagram` and the corresponding
    :class:`DiagramGrid`, produces the Xy-pic representation of the
    diagram.

    The most important method in this class is ``draw``.  Consider the
    following triangle diagram:

    >>> from sympy.categories import Object, NamedMorphism, Diagram
    >>> from sympy.categories import DiagramGrid, XypicDiagramDrawer
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> diagram = Diagram([f, g], {g * f: "unique"})

    To draw this diagram, its objects need to be laid out with a
    :class:`DiagramGrid`::

    >>> grid = DiagramGrid(diagram)

    Finally, the drawing:

    >>> drawer = XypicDiagramDrawer()
    >>> print(drawer.draw(diagram, grid))
    \\xymatrix{
    A \\ar[d]_{g\\circ f} \\ar[r]^{f} & B \\ar[ld]^{g} \\\\
    C &
    }

    For further details see the docstring of this method.

    To control the appearance of the arrows, formatters are used.  The
    dictionary ``arrow_formatters`` maps morphisms to formatter
    functions.  A formatter is accepts an
    :class:`ArrowStringDescription` and is allowed to modify any of
    the arrow properties exposed thereby.  For example, to have all
    morphisms with the property ``unique`` appear as dashed arrows,
    and to have their names prepended with `\\exists !`, the following
    should be done:

    >>> def formatter(astr):
    ...   astr.label = r"\\exists !" + astr.label
    ...   astr.arrow_style = "{-->}"
    >>> drawer.arrow_formatters["unique"] = formatter
    >>> print(drawer.draw(diagram, grid))
    \\xymatrix{
    A \\ar@{-->}[d]_{\\exists !g\\circ f} \\ar[r]^{f} & B \\ar[ld]^{g} \\\\
    C &
    }

    To modify the appearance of all arrows in the diagram, set
    ``default_arrow_formatter``.  For example, to place all morphism
    labels a little bit farther from the arrow head so that they look
    more centred, do as follows:

    >>> def default_formatter(astr):
    ...   astr.label_displacement = "(0.45)"
    >>> drawer.default_arrow_formatter = default_formatter
    >>> print(drawer.draw(diagram, grid))
    \\xymatrix{
    A \\ar@{-->}[d]_(0.45){\\exists !g\\circ f} \\ar[r]^(0.45){f} & B \\ar[ld]^(0.45){g} \\\\
    C &
    }

    In some diagrams some morphisms are drawn as curved arrows.
    Consider the following diagram:

    >>> D = Object("D")
    >>> E = Object("E")
    >>> h = NamedMorphism(D, A, "h")
    >>> k = NamedMorphism(D, B, "k")
    >>> diagram = Diagram([f, g, h, k])
    >>> grid = DiagramGrid(diagram)
    >>> drawer = XypicDiagramDrawer()
    >>> print(drawer.draw(diagram, grid))
    \\xymatrix{
    A \\ar[r]_{f} & B \\ar[d]^{g} & D \\ar[l]^{k} \\ar@/_3mm/[ll]_{h} \\\\
    & C &
    }

    To control how far the morphisms are curved by default, one can
    use the ``unit`` and ``default_curving_amount`` attributes:

    >>> drawer.unit = "cm"
    >>> drawer.default_curving_amount = 1
    >>> print(drawer.draw(diagram, grid))
    \\xymatrix{
    A \\ar[r]_{f} & B \\ar[d]^{g} & D \\ar[l]^{k} \\ar@/_1cm/[ll]_{h} \\\\
    & C &
    }

    In some diagrams, there are multiple curved morphisms between the
    same two objects.  To control by how much the curving changes
    between two such successive morphisms, use
    ``default_curving_step``:

    >>> drawer.default_curving_step = 1
    >>> h1 = NamedMorphism(A, D, "h1")
    >>> diagram = Diagram([f, g, h, k, h1])
    >>> grid = DiagramGrid(diagram)
    >>> print(drawer.draw(diagram, grid))
    \\xymatrix{
    A \\ar[r]_{f} \\ar@/^1cm/[rr]^{h_{1}} & B \\ar[d]^{g} & D \\ar[l]^{k} \\ar@/_2cm/[ll]_{h} \\\\
    & C &
    }

    The default value of ``default_curving_step`` is 4 units.

    See Also
    ========

    draw, ArrowStringDescription
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.unit = 'mm'
        self.default_curving_amount = 3
        self.default_curving_step = 4
        self.arrow_formatters = {}
        self.default_arrow_formatter = None

    @staticmethod
    def _process_loop_morphism(i, j, grid, morphisms_str_info, object_coords):
        if False:
            return 10
        '\n        Produces the information required for constructing the string\n        representation of a loop morphism.  This function is invoked\n        from ``_process_morphism``.\n\n        See Also\n        ========\n\n        _process_morphism\n        '
        curving = ''
        label_pos = '^'
        looping_start = ''
        looping_end = ''
        quadrant = [0, 0, 0, 0]
        obj = grid[i, j]
        for (m, m_str_info) in morphisms_str_info.items():
            if m.domain == obj and m.codomain == obj:
                (l_s, l_e) = (m_str_info.looping_start, m_str_info.looping_end)
                if (l_s, l_e) == ('r', 'u'):
                    quadrant[0] += 1
                elif (l_s, l_e) == ('u', 'l'):
                    quadrant[1] += 1
                elif (l_s, l_e) == ('l', 'd'):
                    quadrant[2] += 1
                elif (l_s, l_e) == ('d', 'r'):
                    quadrant[3] += 1
                continue
            if m.domain == obj:
                (end_i, end_j) = object_coords[m.codomain]
                goes_out = True
            elif m.codomain == obj:
                (end_i, end_j) = object_coords[m.domain]
                goes_out = False
            else:
                continue
            d_i = end_i - i
            d_j = end_j - j
            m_curving = m_str_info.curving
            if d_i != 0 and d_j != 0:
                if d_i > 0 and d_j > 0:
                    quadrant[0] += 1
                elif d_i > 0 and d_j < 0:
                    quadrant[1] += 1
                elif d_i < 0 and d_j < 0:
                    quadrant[2] += 1
                elif d_i < 0 and d_j > 0:
                    quadrant[3] += 1
            elif d_i == 0:
                if d_j > 0:
                    if goes_out:
                        upper_quadrant = 0
                        lower_quadrant = 3
                    else:
                        upper_quadrant = 3
                        lower_quadrant = 0
                elif goes_out:
                    upper_quadrant = 2
                    lower_quadrant = 1
                else:
                    upper_quadrant = 1
                    lower_quadrant = 2
                if m_curving:
                    if m_curving == '^':
                        quadrant[upper_quadrant] += 1
                    elif m_curving == '_':
                        quadrant[lower_quadrant] += 1
                else:
                    quadrant[upper_quadrant] += 1
                    quadrant[lower_quadrant] += 1
            elif d_j == 0:
                if d_i < 0:
                    if goes_out:
                        left_quadrant = 1
                        right_quadrant = 0
                    else:
                        left_quadrant = 0
                        right_quadrant = 1
                elif goes_out:
                    left_quadrant = 3
                    right_quadrant = 2
                else:
                    left_quadrant = 2
                    right_quadrant = 3
                if m_curving:
                    if m_curving == '^':
                        quadrant[left_quadrant] += 1
                    elif m_curving == '_':
                        quadrant[right_quadrant] += 1
                else:
                    quadrant[left_quadrant] += 1
                    quadrant[right_quadrant] += 1
        freest_quadrant = 0
        for i in range(4):
            if quadrant[i] < quadrant[freest_quadrant]:
                freest_quadrant = i
        (looping_start, looping_end) = [('r', 'u'), ('u', 'l'), ('l', 'd'), ('d', 'r')][freest_quadrant]
        return (curving, label_pos, looping_start, looping_end)

    @staticmethod
    def _process_horizontal_morphism(i, j, target_j, grid, morphisms_str_info, object_coords):
        if False:
            i = 10
            return i + 15
        '\n        Produces the information required for constructing the string\n        representation of a horizontal morphism.  This function is\n        invoked from ``_process_morphism``.\n\n        See Also\n        ========\n\n        _process_morphism\n        '
        backwards = False
        start = j
        end = target_j
        if end < start:
            (start, end) = (end, start)
            backwards = True
        up = []
        down = []
        straight_horizontal = []
        for k in range(start + 1, end):
            obj = grid[i, k]
            if not obj:
                continue
            for m in morphisms_str_info:
                if m.domain == obj:
                    (end_i, end_j) = object_coords[m.codomain]
                elif m.codomain == obj:
                    (end_i, end_j) = object_coords[m.domain]
                else:
                    continue
                if end_i > i:
                    down.append(m)
                elif end_i < i:
                    up.append(m)
                elif not morphisms_str_info[m].curving:
                    straight_horizontal.append(m)
        if len(up) < len(down):
            if backwards:
                curving = '_'
                label_pos = '_'
            else:
                curving = '^'
                label_pos = '^'
            for m in straight_horizontal:
                (i1, j1) = object_coords[m.domain]
                (i2, j2) = object_coords[m.codomain]
                m_str_info = morphisms_str_info[m]
                if j1 < j2:
                    m_str_info.label_position = '_'
                else:
                    m_str_info.label_position = '^'
                m_str_info.forced_label_position = True
        else:
            if backwards:
                curving = '^'
                label_pos = '^'
            else:
                curving = '_'
                label_pos = '_'
            for m in straight_horizontal:
                (i1, j1) = object_coords[m.domain]
                (i2, j2) = object_coords[m.codomain]
                m_str_info = morphisms_str_info[m]
                if j1 < j2:
                    m_str_info.label_position = '^'
                else:
                    m_str_info.label_position = '_'
                m_str_info.forced_label_position = True
        return (curving, label_pos)

    @staticmethod
    def _process_vertical_morphism(i, j, target_i, grid, morphisms_str_info, object_coords):
        if False:
            return 10
        '\n        Produces the information required for constructing the string\n        representation of a vertical morphism.  This function is\n        invoked from ``_process_morphism``.\n\n        See Also\n        ========\n\n        _process_morphism\n        '
        backwards = False
        start = i
        end = target_i
        if end < start:
            (start, end) = (end, start)
            backwards = True
        left = []
        right = []
        straight_vertical = []
        for k in range(start + 1, end):
            obj = grid[k, j]
            if not obj:
                continue
            for m in morphisms_str_info:
                if m.domain == obj:
                    (end_i, end_j) = object_coords[m.codomain]
                elif m.codomain == obj:
                    (end_i, end_j) = object_coords[m.domain]
                else:
                    continue
                if end_j > j:
                    right.append(m)
                elif end_j < j:
                    left.append(m)
                elif not morphisms_str_info[m].curving:
                    straight_vertical.append(m)
        if len(left) < len(right):
            if backwards:
                curving = '^'
                label_pos = '^'
            else:
                curving = '_'
                label_pos = '_'
            for m in straight_vertical:
                (i1, j1) = object_coords[m.domain]
                (i2, j2) = object_coords[m.codomain]
                m_str_info = morphisms_str_info[m]
                if i1 < i2:
                    m_str_info.label_position = '^'
                else:
                    m_str_info.label_position = '_'
                m_str_info.forced_label_position = True
        else:
            if backwards:
                curving = '_'
                label_pos = '_'
            else:
                curving = '^'
                label_pos = '^'
            for m in straight_vertical:
                (i1, j1) = object_coords[m.domain]
                (i2, j2) = object_coords[m.codomain]
                m_str_info = morphisms_str_info[m]
                if i1 < i2:
                    m_str_info.label_position = '_'
                else:
                    m_str_info.label_position = '^'
                m_str_info.forced_label_position = True
        return (curving, label_pos)

    def _process_morphism(self, diagram, grid, morphism, object_coords, morphisms, morphisms_str_info):
        if False:
            print('Hello World!')
        '\n        Given the required information, produces the string\n        representation of ``morphism``.\n        '

        def repeat_string_cond(times, str_gt, str_lt):
            if False:
                return 10
            '\n            If ``times > 0``, repeats ``str_gt`` ``times`` times.\n            Otherwise, repeats ``str_lt`` ``-times`` times.\n            '
            if times > 0:
                return str_gt * times
            else:
                return str_lt * -times

        def count_morphisms_undirected(A, B):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Counts how many processed morphisms there are between the\n            two supplied objects.\n            '
            return len([m for m in morphisms_str_info if {m.domain, m.codomain} == {A, B}])

        def count_morphisms_filtered(dom, cod, curving):
            if False:
                print('Hello World!')
            '\n            Counts the processed morphisms which go out of ``dom``\n            into ``cod`` with curving ``curving``.\n            '
            return len([m for (m, m_str_info) in morphisms_str_info.items() if (m.domain, m.codomain) == (dom, cod) and m_str_info.curving == curving])
        (i, j) = object_coords[morphism.domain]
        (target_i, target_j) = object_coords[morphism.codomain]
        delta_i = target_i - i
        delta_j = target_j - j
        vertical_direction = repeat_string_cond(delta_i, 'd', 'u')
        horizontal_direction = repeat_string_cond(delta_j, 'r', 'l')
        curving = ''
        label_pos = '^'
        looping_start = ''
        looping_end = ''
        if delta_i == 0 and delta_j == 0:
            (curving, label_pos, looping_start, looping_end) = XypicDiagramDrawer._process_loop_morphism(i, j, grid, morphisms_str_info, object_coords)
        elif delta_i == 0 and abs(j - target_j) > 1:
            (curving, label_pos) = XypicDiagramDrawer._process_horizontal_morphism(i, j, target_j, grid, morphisms_str_info, object_coords)
        elif delta_j == 0 and abs(i - target_i) > 1:
            (curving, label_pos) = XypicDiagramDrawer._process_vertical_morphism(i, j, target_i, grid, morphisms_str_info, object_coords)
        count = count_morphisms_undirected(morphism.domain, morphism.codomain)
        curving_amount = ''
        if curving:
            curving_amount = self.default_curving_amount + count * self.default_curving_step
        elif count:
            curving = '^'
            filtered_morphisms = count_morphisms_filtered(morphism.domain, morphism.codomain, curving)
            curving_amount = self.default_curving_amount + filtered_morphisms * self.default_curving_step
        morphism_name = ''
        if isinstance(morphism, IdentityMorphism):
            morphism_name = 'id_{%s}' + latex(grid[i, j])
        elif isinstance(morphism, CompositeMorphism):
            component_names = [latex(Symbol(component.name)) for component in morphism.components]
            component_names.reverse()
            morphism_name = '\\circ '.join(component_names)
        elif isinstance(morphism, NamedMorphism):
            morphism_name = latex(Symbol(morphism.name))
        return ArrowStringDescription(self.unit, curving, curving_amount, looping_start, looping_end, horizontal_direction, vertical_direction, label_pos, morphism_name)

    @staticmethod
    def _check_free_space_horizontal(dom_i, dom_j, cod_j, grid):
        if False:
            print('Hello World!')
        '\n        For a horizontal morphism, checks whether there is free space\n        (i.e., space not occupied by any objects) above the morphism\n        or below it.\n        '
        if dom_j < cod_j:
            (start, end) = (dom_j, cod_j)
            backwards = False
        else:
            (start, end) = (cod_j, dom_j)
            backwards = True
        if dom_i == 0:
            free_up = True
        else:
            free_up = all((grid[dom_i - 1, j] for j in range(start, end + 1)))
        if dom_i == grid.height - 1:
            free_down = True
        else:
            free_down = not any((grid[dom_i + 1, j] for j in range(start, end + 1)))
        return (free_up, free_down, backwards)

    @staticmethod
    def _check_free_space_vertical(dom_i, cod_i, dom_j, grid):
        if False:
            print('Hello World!')
        '\n        For a vertical morphism, checks whether there is free space\n        (i.e., space not occupied by any objects) to the left of the\n        morphism or to the right of it.\n        '
        if dom_i < cod_i:
            (start, end) = (dom_i, cod_i)
            backwards = False
        else:
            (start, end) = (cod_i, dom_i)
            backwards = True
        if dom_j == 0:
            free_left = True
        else:
            free_left = not any((grid[i, dom_j - 1] for i in range(start, end + 1)))
        if dom_j == grid.width - 1:
            free_right = True
        else:
            free_right = not any((grid[i, dom_j + 1] for i in range(start, end + 1)))
        return (free_left, free_right, backwards)

    @staticmethod
    def _check_free_space_diagonal(dom_i, cod_i, dom_j, cod_j, grid):
        if False:
            i = 10
            return i + 15
        '\n        For a diagonal morphism, checks whether there is free space\n        (i.e., space not occupied by any objects) above the morphism\n        or below it.\n        '

        def abs_xrange(start, end):
            if False:
                return 10
            if start < end:
                return range(start, end + 1)
            else:
                return range(end, start + 1)
        if dom_i < cod_i and dom_j < cod_j:
            (start_i, start_j) = (dom_i, dom_j)
            (end_i, end_j) = (cod_i, cod_j)
            backwards = False
        elif dom_i > cod_i and dom_j > cod_j:
            (start_i, start_j) = (cod_i, cod_j)
            (end_i, end_j) = (dom_i, dom_j)
            backwards = True
        if dom_i < cod_i and dom_j > cod_j:
            (start_i, start_j) = (dom_i, dom_j)
            (end_i, end_j) = (cod_i, cod_j)
            backwards = True
        elif dom_i > cod_i and dom_j < cod_j:
            (start_i, start_j) = (cod_i, cod_j)
            (end_i, end_j) = (dom_i, dom_j)
            backwards = False
        alpha = float(end_i - start_i) / (end_j - start_j)
        free_up = True
        free_down = True
        for i in abs_xrange(start_i, end_i):
            if not free_up and (not free_down):
                break
            for j in abs_xrange(start_j, end_j):
                if not free_up and (not free_down):
                    break
                if (i, j) == (start_i, start_j):
                    continue
                if j == start_j:
                    alpha1 = 'inf'
                else:
                    alpha1 = float(i - start_i) / (j - start_j)
                if grid[i, j]:
                    if alpha1 == 'inf' or abs(alpha1) > abs(alpha):
                        free_down = False
                    elif abs(alpha1) < abs(alpha):
                        free_up = False
        return (free_up, free_down, backwards)

    def _push_labels_out(self, morphisms_str_info, grid, object_coords):
        if False:
            for i in range(10):
                print('nop')
        '\n        For all straight morphisms which form the visual boundary of\n        the laid out diagram, puts their labels on their outer sides.\n        '

        def set_label_position(free1, free2, pos1, pos2, backwards, m_str_info):
            if False:
                while True:
                    i = 10
            '\n            Given the information about room available to one side and\n            to the other side of a morphism (``free1`` and ``free2``),\n            sets the position of the morphism label in such a way that\n            it is on the freer side.  This latter operations involves\n            choice between ``pos1`` and ``pos2``, taking ``backwards``\n            in consideration.\n\n            Thus this function will do nothing if either both ``free1\n            == True`` and ``free2 == True`` or both ``free1 == False``\n            and ``free2 == False``.  In either case, choosing one side\n            over the other presents no advantage.\n            '
            if backwards:
                (pos1, pos2) = (pos2, pos1)
            if free1 and (not free2):
                m_str_info.label_position = pos1
            elif free2 and (not free1):
                m_str_info.label_position = pos2
        for (m, m_str_info) in morphisms_str_info.items():
            if m_str_info.curving or m_str_info.forced_label_position:
                continue
            if m.domain == m.codomain:
                continue
            (dom_i, dom_j) = object_coords[m.domain]
            (cod_i, cod_j) = object_coords[m.codomain]
            if dom_i == cod_i:
                (free_up, free_down, backwards) = XypicDiagramDrawer._check_free_space_horizontal(dom_i, dom_j, cod_j, grid)
                set_label_position(free_up, free_down, '^', '_', backwards, m_str_info)
            elif dom_j == cod_j:
                (free_left, free_right, backwards) = XypicDiagramDrawer._check_free_space_vertical(dom_i, cod_i, dom_j, grid)
                set_label_position(free_left, free_right, '_', '^', backwards, m_str_info)
            else:
                (free_up, free_down, backwards) = XypicDiagramDrawer._check_free_space_diagonal(dom_i, cod_i, dom_j, cod_j, grid)
                set_label_position(free_up, free_down, '^', '_', backwards, m_str_info)

    @staticmethod
    def _morphism_sort_key(morphism, object_coords):
        if False:
            for i in range(10):
                print('nop')
        '\n        Provides a morphism sorting key such that horizontal or\n        vertical morphisms between neighbouring objects come\n        first, then horizontal or vertical morphisms between more\n        far away objects, and finally, all other morphisms.\n        '
        (i, j) = object_coords[morphism.domain]
        (target_i, target_j) = object_coords[morphism.codomain]
        if morphism.domain == morphism.codomain:
            return (3, 0, default_sort_key(morphism))
        if target_i == i:
            return (1, abs(target_j - j), default_sort_key(morphism))
        if target_j == j:
            return (1, abs(target_i - i), default_sort_key(morphism))
        return (2, 0, default_sort_key(morphism))

    @staticmethod
    def _build_xypic_string(diagram, grid, morphisms, morphisms_str_info, diagram_format):
        if False:
            while True:
                i = 10
        '\n        Given a collection of :class:`ArrowStringDescription`\n        describing the morphisms of a diagram and the object layout\n        information of a diagram, produces the final Xy-pic picture.\n        '
        object_morphisms = {}
        for obj in diagram.objects:
            object_morphisms[obj] = []
        for morphism in morphisms:
            object_morphisms[morphism.domain].append(morphism)
        result = '\\xymatrix%s{\n' % diagram_format
        for i in range(grid.height):
            for j in range(grid.width):
                obj = grid[i, j]
                if obj:
                    result += latex(obj) + ' '
                    morphisms_to_draw = object_morphisms[obj]
                    for morphism in morphisms_to_draw:
                        result += str(morphisms_str_info[morphism]) + ' '
                if j < grid.width - 1:
                    result += '& '
            if i < grid.height - 1:
                result += '\\\\'
            result += '\n'
        result += '}\n'
        return result

    def draw(self, diagram, grid, masked=None, diagram_format=''):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the Xy-pic representation of ``diagram`` laid out in\n        ``grid``.\n\n        Consider the following simple triangle diagram.\n\n        >>> from sympy.categories import Object, NamedMorphism, Diagram\n        >>> from sympy.categories import DiagramGrid, XypicDiagramDrawer\n        >>> A = Object("A")\n        >>> B = Object("B")\n        >>> C = Object("C")\n        >>> f = NamedMorphism(A, B, "f")\n        >>> g = NamedMorphism(B, C, "g")\n        >>> diagram = Diagram([f, g], {g * f: "unique"})\n\n        To draw this diagram, its objects need to be laid out with a\n        :class:`DiagramGrid`::\n\n        >>> grid = DiagramGrid(diagram)\n\n        Finally, the drawing:\n\n        >>> drawer = XypicDiagramDrawer()\n        >>> print(drawer.draw(diagram, grid))\n        \\xymatrix{\n        A \\ar[d]_{g\\circ f} \\ar[r]^{f} & B \\ar[ld]^{g} \\\\\n        C &\n        }\n\n        The argument ``masked`` can be used to skip morphisms in the\n        presentation of the diagram:\n\n        >>> print(drawer.draw(diagram, grid, masked=[g * f]))\n        \\xymatrix{\n        A \\ar[r]^{f} & B \\ar[ld]^{g} \\\\\n        C &\n        }\n\n        Finally, the ``diagram_format`` argument can be used to\n        specify the format string of the diagram.  For example, to\n        increase the spacing by 1 cm, proceeding as follows:\n\n        >>> print(drawer.draw(diagram, grid, diagram_format="@+1cm"))\n        \\xymatrix@+1cm{\n        A \\ar[d]_{g\\circ f} \\ar[r]^{f} & B \\ar[ld]^{g} \\\\\n        C &\n        }\n\n        '
        if not masked:
            morphisms_props = grid.morphisms
        else:
            morphisms_props = {}
            for (m, props) in grid.morphisms.items():
                if m in masked:
                    continue
                morphisms_props[m] = props
        object_coords = {}
        for i in range(grid.height):
            for j in range(grid.width):
                if grid[i, j]:
                    object_coords[grid[i, j]] = (i, j)
        morphisms = sorted(morphisms_props, key=lambda m: XypicDiagramDrawer._morphism_sort_key(m, object_coords))
        morphisms_str_info = {}
        for morphism in morphisms:
            string_description = self._process_morphism(diagram, grid, morphism, object_coords, morphisms, morphisms_str_info)
            if self.default_arrow_formatter:
                self.default_arrow_formatter(string_description)
            for prop in morphisms_props[morphism]:
                if prop.name in self.arrow_formatters:
                    formatter = self.arrow_formatters[prop.name]
                    formatter(string_description)
            morphisms_str_info[morphism] = string_description
        self._push_labels_out(morphisms_str_info, grid, object_coords)
        return XypicDiagramDrawer._build_xypic_string(diagram, grid, morphisms, morphisms_str_info, diagram_format)

def xypic_draw_diagram(diagram, masked=None, diagram_format='', groups=None, **hints):
    if False:
        for i in range(10):
            print('nop')
    '\n    Provides a shortcut combining :class:`DiagramGrid` and\n    :class:`XypicDiagramDrawer`.  Returns an Xy-pic presentation of\n    ``diagram``.  The argument ``masked`` is a list of morphisms which\n    will be not be drawn.  The argument ``diagram_format`` is the\n    format string inserted after "\\xymatrix".  ``groups`` should be a\n    set of logical groups.  The ``hints`` will be passed directly to\n    the constructor of :class:`DiagramGrid`.\n\n    For more information about the arguments, see the docstrings of\n    :class:`DiagramGrid` and ``XypicDiagramDrawer.draw``.\n\n    Examples\n    ========\n\n    >>> from sympy.categories import Object, NamedMorphism, Diagram\n    >>> from sympy.categories import xypic_draw_diagram\n    >>> A = Object("A")\n    >>> B = Object("B")\n    >>> C = Object("C")\n    >>> f = NamedMorphism(A, B, "f")\n    >>> g = NamedMorphism(B, C, "g")\n    >>> diagram = Diagram([f, g], {g * f: "unique"})\n    >>> print(xypic_draw_diagram(diagram))\n    \\xymatrix{\n    A \\ar[d]_{g\\circ f} \\ar[r]^{f} & B \\ar[ld]^{g} \\\\\n    C &\n    }\n\n    See Also\n    ========\n\n    XypicDiagramDrawer, DiagramGrid\n    '
    grid = DiagramGrid(diagram, groups, **hints)
    drawer = XypicDiagramDrawer()
    return drawer.draw(diagram, grid, masked, diagram_format)

@doctest_depends_on(exe=('latex', 'dvipng'), modules=('pyglet',))
def preview_diagram(diagram, masked=None, diagram_format='', groups=None, output='png', viewer=None, euler=True, **hints):
    if False:
        print('Hello World!')
    '\n    Combines the functionality of ``xypic_draw_diagram`` and\n    ``sympy.printing.preview``.  The arguments ``masked``,\n    ``diagram_format``, ``groups``, and ``hints`` are passed to\n    ``xypic_draw_diagram``, while ``output``, ``viewer, and ``euler``\n    are passed to ``preview``.\n\n    Examples\n    ========\n\n    >>> from sympy.categories import Object, NamedMorphism, Diagram\n    >>> from sympy.categories import preview_diagram\n    >>> A = Object("A")\n    >>> B = Object("B")\n    >>> C = Object("C")\n    >>> f = NamedMorphism(A, B, "f")\n    >>> g = NamedMorphism(B, C, "g")\n    >>> d = Diagram([f, g], {g * f: "unique"})\n    >>> preview_diagram(d)\n\n    See Also\n    ========\n\n    XypicDiagramDrawer\n    '
    from sympy.printing import preview
    latex_output = xypic_draw_diagram(diagram, masked, diagram_format, groups, **hints)
    preview(latex_output, output, viewer, euler, ('xypic',))