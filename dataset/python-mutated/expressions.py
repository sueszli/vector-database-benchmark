""" Represent array expressions to be computed on the client (browser) side
by BokehJS.

Expression models are useful as ``DataSpec`` values when it is desired that
the array values be computed in the browser:

.. code-block:: python

    p.circle(x={'expr': some_expression}, ...)

or using the ``expr`` convenience function:

.. code-block:: python

    from bokeh.core.properties import expr

    p.circle(x=expr(some_expression), ...)

In this case, the values of the ``x`` coordinates will be computed in the
browser by the JavaScript implementation of ``some_expression`` using a
``ColumnDataSource`` as input.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from math import inf
from ..core.enums import Direction
from ..core.has_props import abstract
from ..core.properties import AngleSpec, AnyRef, Bool, Dict, Enum, Float, Instance, Nullable, NumberSpec, Required, Seq, String, field
from ..model import Model
__all__ = ('CumSum', 'CustomJSExpr', 'Expression', 'PolarTransform', 'Stack')

@abstract
class Expression(Model):
    """ Base class for ``Expression`` models that represent a computation
    to be carried out on the client-side.

    JavaScript implementations should implement the following methods:

    .. code-block

        v_compute(source: ColumnarDataSource): Arrayable {
            # compute and return array of values
        }

    .. note::
        If you wish for results to be cached per source and updated only if
        the source changes, implement ``_v_compute: (source)`` instead.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)

class CustomJSExpr(Expression):
    """ Evaluate a JavaScript function/generator.

    .. warning::
        The explicit purpose of this Bokeh Model is to embed *raw JavaScript
        code* for a browser to execute. If any part of the code is derived
        from untrusted user inputs, then you must take appropriate care to
        sanitize the user input prior to passing to Bokeh.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    args = Dict(String, AnyRef, help="\n    A mapping of names to Python objects. In particular those can be bokeh's models.\n    These objects are made available to the callback's code snippet as the values of\n    named parameters to the callback. There is no need to manually include the data\n    source of the associated glyph renderer, as it is available within the scope of\n    the code via `this` keyword (e.g. `this.data` will give access to raw data).\n    ")
    code = String(default='', help='\n    A snippet of JavaScript code to execute in the browser. The code is made into\n    the body of a generator function, and all of of the named objects in ``args``\n    are available as parameters that the code can use. One can either return an\n    array-like object (array, typed array, nd-array), an iterable (which will\n    be converted to an array) or a scalar value (which will be converted into\n    an array of an appropriate length), or alternatively yield values that will\n    be collected into an array.\n    ')

class CumSum(Expression):
    """ An expression for generating arrays by cumulatively summing a single
    column from a ``ColumnDataSource``.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    field = Required(String, help='\n    The name of a ``ColumnDataSource`` column to cumulatively sum for new values.\n    ')
    include_zero = Bool(default=False, help="\n    Whether to include zero at the start of the result. Note that the length\n    of the result is always the same as the input column. Therefore if this\n    property is True, then the last value of the column will not be included\n    in the sum.\n\n    .. code-block:: python\n\n        source = ColumnDataSource(data=dict(foo=[1, 2, 3, 4]))\n\n        CumSum(field='foo')\n        # -> [1, 3, 6, 10]\n\n        CumSum(field='foo', include_zero=True)\n        # -> [0, 1, 3, 6]\n\n    ")

class Stack(Expression):
    """ An expression for generating arrays by summing different columns from
    a ``ColumnDataSource``.

    This expression is useful for implementing stacked bar charts at a low
    level.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    fields = Seq(String, default=[], help="\n    A sequence of fields from a ``ColumnDataSource`` to sum (elementwise). For\n    example:\n\n    .. code-block:: python\n\n        Stack(fields=['sales', 'marketing'])\n\n    Will compute an array of values (in the browser) by adding the elements\n    of the ``'sales'`` and ``'marketing'`` columns of a data source.\n    ")

@abstract
class ScalarExpression(Model):
    """ Base class for for scalar expressions. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

class Minimum(ScalarExpression):
    """ Computes minimum value of a data source's column. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    field = Required(String)
    initial = Nullable(Float, default=inf)

class Maximum(ScalarExpression):
    """ Computes maximum value of a data source's column. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    field = Required(String)
    initial = Nullable(Float, default=-inf)

@abstract
class CoordinateTransform(Expression):
    """ Base class for coordinate transforms. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

    @property
    def x(self):
        if False:
            print('Hello World!')
        return XComponent(transform=self)

    @property
    def y(self):
        if False:
            i = 10
            return i + 15
        return YComponent(transform=self)

class PolarTransform(CoordinateTransform):
    """ Transform from polar to cartesian coordinates. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    radius = NumberSpec(default=field('radius'), help='\n    The radial coordinate (i.e. the distance from the origin).\n\n    Negative radius is allowed, which is equivalent to using positive radius\n    and changing ``direction`` to the opposite value.\n    ')
    angle = AngleSpec(default=field('angle'), help='\n    The angular coordinate (i.e. the angle from the reference axis).\n    ')
    direction = Enum(Direction, default=Direction.anticlock, help='\n    Whether ``angle`` measures clockwise or anti-clockwise from the reference axis.\n    ')

@abstract
class XYComponent(Expression):
    """ Base class for bi-variate expressions. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    transform = Instance(CoordinateTransform)

class XComponent(XYComponent):
    """ X-component of a coordinate system transform to cartesian coordinates. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)

class YComponent(XYComponent):
    """ Y-component of a coordinate system transform to cartesian coordinates. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)