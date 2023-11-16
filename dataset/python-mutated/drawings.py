"""
Drawing objects for timeline drawer.

Drawing objects play two important roles:
    - Allowing unittests of visualization module. Usually it is hard for image files to be tested.
    - Removing program parser from each plotter interface. We can easily add new plotter.

This module is based on the structure of matplotlib as it is the primary plotter
of the timeline drawer. However this interface is agnostic to the actual plotter.

Design concept
~~~~~~~~~~~~~~
When we think about dynamically updating drawings, it will be most efficient to
update only the changed properties of drawings rather than regenerating entirely from scratch.
Thus the core :py:class:`~qiskit.visualization.timeline.core.DrawerCanvas` generates
all possible drawings in the beginning and then the canvas instance manages
visibility of each drawing according to the end-user request.

Data key
~~~~~~~~
In the abstract class ``ElementaryData`` common attributes to represent a drawing are
specified. In addition, drawings have the `data_key` property that returns an
unique hash of the object for comparison.
This key is generated from a data type, the location of the drawing in the canvas,
and associated qubit or classical bit objects.
See py:mod:`qiskit.visualization.timeline.types` for detail on the data type.
If a data key cannot distinguish two independent objects, you need to add a new data type.
The data key may be used in the plotter interface to identify the object.

Drawing objects
~~~~~~~~~~~~~~~
To support not only `matplotlib` but also multiple plotters, those drawings should be
universal and designed without strong dependency on modules in `matplotlib`.
This means drawings that represent primitive geometries are preferred.
It should be noted that there will be no unittest for each plotter API, which takes
drawings and outputs image data, we should avoid adding a complicated geometry
that has a context of the scheduled circuit program.

For example, a two qubit scheduled gate may be drawn by two rectangles that represent
time occupation of two quantum registers during the gate along with a line connecting
these rectangles to identify the pair. This shape can be represented with
two box-type objects with one line-type object instead of defining a new object dedicated
to the two qubit gate. As many plotters don't support an API that visualizes such
a linked-box shape, if we introduce such complex drawings and write a
custom wrapper function on top of the existing API,
it could be difficult to prevent bugs with the CI tools due to lack of
the effective unittest for image data.

Link between gates
~~~~~~~~~~~~~~~~~~
The ``GateLinkData`` is the special subclass of drawing that represents
a link between bits. Usually objects are associated to the specific bit,
but ``GateLinkData`` can be associated with multiple bits to illustrate relationship
between quantum or classical bits during a gate operation.
"""
from abc import ABC
from enum import Enum
from typing import Optional, Dict, Any, List, Union
import numpy as np
from qiskit import circuit
from qiskit.visualization.timeline import types
from qiskit.visualization.exceptions import VisualizationError

class ElementaryData(ABC):
    """Base class of the scheduled circuit visualization object.

    Note that drawings are mutable.
    """
    __hash__ = None

    def __init__(self, data_type: Union[str, Enum], xvals: Union[np.ndarray, List[types.Coordinate]], yvals: Union[np.ndarray, List[types.Coordinate]], bits: Optional[Union[types.Bits, List[types.Bits]]]=None, meta: Optional[Dict[str, Any]]=None, styles: Optional[Dict[str, Any]]=None):
        if False:
            print('Hello World!')
        'Create new drawing.\n\n        Args:\n            data_type: String representation of this drawing.\n            xvals: Series of horizontal coordinate that the object is drawn.\n            yvals: Series of vertical coordinate that the object is drawn.\n            bits: Qubit or Clbit object bound to this drawing.\n            meta: Meta data dictionary of the object.\n            styles: Style keyword args of the object. This conforms to `matplotlib`.\n        '
        if bits and isinstance(bits, (circuit.Qubit, circuit.Clbit)):
            bits = [bits]
        if isinstance(data_type, Enum):
            data_type = data_type.value
        self.data_type = str(data_type)
        self.xvals = xvals
        self.yvals = yvals
        self.bits = bits
        self.meta = meta
        self.styles = styles

    @property
    def data_key(self):
        if False:
            print('Hello World!')
        'Return unique hash of this object.'
        return str(hash((self.__class__.__name__, self.data_type, tuple(self.bits), tuple(self.xvals), tuple(self.yvals))))

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.__class__.__name__}(type={self.data_type}, key={self.data_key})'

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, self.__class__) and self.data_key == other.data_key

class LineData(ElementaryData):
    """Drawing object that represents line shape."""

    def __init__(self, data_type: Union[str, Enum], xvals: Union[np.ndarray, List[types.Coordinate]], yvals: Union[np.ndarray, List[types.Coordinate]], bit: types.Bits, meta: Dict[str, Any]=None, styles: Dict[str, Any]=None):
        if False:
            while True:
                i = 10
        'Create new line.\n\n        Args:\n            data_type: String representation of this drawing.\n            xvals: Series of horizontal coordinate that the object is drawn.\n            yvals: Series of vertical coordinate that the object is drawn.\n            bit: Bit associated to this object.\n            meta: Meta data dictionary of the object.\n            styles: Style keyword args of the object. This conforms to `matplotlib`.\n        '
        super().__init__(data_type=data_type, xvals=xvals, yvals=yvals, bits=bit, meta=meta, styles=styles)

class BoxData(ElementaryData):
    """Drawing object that represents box shape."""

    def __init__(self, data_type: Union[str, Enum], xvals: Union[np.ndarray, List[types.Coordinate]], yvals: Union[np.ndarray, List[types.Coordinate]], bit: types.Bits, meta: Dict[str, Any]=None, styles: Dict[str, Any]=None):
        if False:
            for i in range(10):
                print('nop')
        'Create new box.\n\n        Args:\n            data_type: String representation of this drawing.\n            xvals: Left and right coordinate that the object is drawn.\n            yvals: Top and bottom coordinate that the object is drawn.\n            bit: Bit associated to this object.\n            meta: Meta data dictionary of the object.\n            styles: Style keyword args of the object. This conforms to `matplotlib`.\n\n        Raises:\n            VisualizationError: When number of data points are not equals to 2.\n        '
        if len(xvals) != 2 or len(yvals) != 2:
            raise VisualizationError('Length of data points are not equals to 2.')
        super().__init__(data_type=data_type, xvals=xvals, yvals=yvals, bits=bit, meta=meta, styles=styles)

class TextData(ElementaryData):
    """Drawing object that represents a text on canvas."""

    def __init__(self, data_type: Union[str, Enum], xval: types.Coordinate, yval: types.Coordinate, bit: types.Bits, text: str, latex: Optional[str]=None, meta: Dict[str, Any]=None, styles: Dict[str, Any]=None):
        if False:
            print('Hello World!')
        'Create new text.\n\n        Args:\n            data_type: String representation of this drawing.\n            xval: Horizontal coordinate that the object is drawn.\n            yval: Vertical coordinate that the object is drawn.\n            bit: Bit associated to this object.\n            text: A string to draw on the canvas.\n            latex: If set this string is used instead of `text`.\n            meta: Meta data dictionary of the object.\n            styles: Style keyword args of the object. This conforms to `matplotlib`.\n        '
        self.text = text
        self.latex = latex
        super().__init__(data_type=data_type, xvals=[xval], yvals=[yval], bits=bit, meta=meta, styles=styles)

class GateLinkData(ElementaryData):
    """A special drawing data type that represents bit link of multi-bit gates.

    Note this object takes multiple bits and dedicates them to the bit link.
    This may appear as a line on the canvas.
    """

    def __init__(self, xval: types.Coordinate, bits: List[types.Bits], styles: Dict[str, Any]=None):
        if False:
            print('Hello World!')
        'Create new bit link.\n\n        Args:\n            xval: Horizontal coordinate that the object is drawn.\n            bits: Bit associated to this object.\n            styles: Style keyword args of the object. This conforms to `matplotlib`.\n        '
        super().__init__(data_type=types.LineType.GATE_LINK, xvals=[xval], yvals=[0], bits=bits, meta=None, styles=styles)