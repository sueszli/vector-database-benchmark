from __future__ import annotations
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Iterable, cast
import numpy as np
import numpy.typing as npt
import pyarrow as pa
if TYPE_CHECKING:
    from .._log import ComponentBatchLike
    from . import ViewCoordinates, ViewCoordinatesArrayLike

class ViewCoordinatesExt:
    """Extension for [ViewCoordinates][rerun.components.ViewCoordinates]."""

    class ViewDir(IntEnum):
        Up = 1
        Down = 2
        Right = 3
        Left = 4
        Forward = 5
        Back = 6

    @staticmethod
    def coordinates__field_converter_override(data: npt.ArrayLike) -> npt.NDArray[np.uint8]:
        if False:
            for i in range(10):
                print('nop')
        coordinates = np.asarray(data, dtype=np.uint8)
        if coordinates.shape != (3,):
            raise ValueError(f'ViewCoordinates must be a 3-element array. Got: {coordinates.shape}')
        return coordinates

    @staticmethod
    def native_to_pa_array_override(data: ViewCoordinatesArrayLike, data_type: pa.DataType) -> pa.Array:
        if False:
            while True:
                i = 10
        from . import ViewCoordinates, ViewCoordinatesLike
        if isinstance(data, ViewCoordinates):
            data = [data.coordinates]
        elif hasattr(data, '__len__') and len(data) > 0 and isinstance(data[0], ViewCoordinates):
            data = [d.coordinates for d in data]
        else:
            data = cast(ViewCoordinatesLike, data)
            try:
                data = [ViewCoordinates(data).coordinates]
            except ValueError:
                data = [ViewCoordinates(d).coordinates for d in data]
        data = np.asarray(data, dtype=np.uint8)
        if len(data.shape) != 2 or data.shape[1] != 3:
            raise ValueError(f'ViewCoordinates must be a 3-element array. Got: {data.shape}')
        data = data.flatten()
        for value in data:
            if value not in range(1, 7):
                raise ValueError('ViewCoordinates must contain only values in the range [1,6].')
        return pa.FixedSizeListArray.from_arrays(data, type=data_type)

    def as_component_batches(self) -> Iterable[ComponentBatchLike]:
        if False:
            print('Hello World!')
        from ..archetypes import ViewCoordinates
        from ..components import ViewCoordinates as ViewCoordinatesComponent
        return ViewCoordinates(cast(ViewCoordinatesComponent, self)).as_component_batches()

    def num_instances(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 1
    ULF: ViewCoordinates = None
    UFL: ViewCoordinates = None
    LUF: ViewCoordinates = None
    LFU: ViewCoordinates = None
    FUL: ViewCoordinates = None
    FLU: ViewCoordinates = None
    ULB: ViewCoordinates = None
    UBL: ViewCoordinates = None
    LUB: ViewCoordinates = None
    LBU: ViewCoordinates = None
    BUL: ViewCoordinates = None
    BLU: ViewCoordinates = None
    URF: ViewCoordinates = None
    UFR: ViewCoordinates = None
    RUF: ViewCoordinates = None
    RFU: ViewCoordinates = None
    FUR: ViewCoordinates = None
    FRU: ViewCoordinates = None
    URB: ViewCoordinates = None
    UBR: ViewCoordinates = None
    RUB: ViewCoordinates = None
    RBU: ViewCoordinates = None
    BUR: ViewCoordinates = None
    BRU: ViewCoordinates = None
    DLF: ViewCoordinates = None
    DFL: ViewCoordinates = None
    LDF: ViewCoordinates = None
    LFD: ViewCoordinates = None
    FDL: ViewCoordinates = None
    FLD: ViewCoordinates = None
    DLB: ViewCoordinates = None
    DBL: ViewCoordinates = None
    LDB: ViewCoordinates = None
    LBD: ViewCoordinates = None
    BDL: ViewCoordinates = None
    BLD: ViewCoordinates = None
    DRF: ViewCoordinates = None
    DFR: ViewCoordinates = None
    RDF: ViewCoordinates = None
    RFD: ViewCoordinates = None
    FDR: ViewCoordinates = None
    FRD: ViewCoordinates = None
    DRB: ViewCoordinates = None
    DBR: ViewCoordinates = None
    RDB: ViewCoordinates = None
    RBD: ViewCoordinates = None
    BDR: ViewCoordinates = None
    BRD: ViewCoordinates = None
    RIGHT_HAND_X_UP: ViewCoordinates = None
    RIGHT_HAND_X_DOWN: ViewCoordinates = None
    RIGHT_HAND_Y_UP: ViewCoordinates = None
    RIGHT_HAND_Y_DOWN: ViewCoordinates = None
    RIGHT_HAND_Z_UP: ViewCoordinates = None
    RIGHT_HAND_Z_DOWN: ViewCoordinates = None
    LEFT_HAND_X_UP: ViewCoordinates = None
    LEFT_HAND_X_DOWN: ViewCoordinates = None
    LEFT_HAND_Y_UP: ViewCoordinates = None
    LEFT_HAND_Y_DOWN: ViewCoordinates = None
    LEFT_HAND_Z_UP: ViewCoordinates = None
    LEFT_HAND_Z_DOWN: ViewCoordinates = None

    @staticmethod
    def deferred_patch_class(cls: Any) -> None:
        if False:
            print('Hello World!')
        cls.ULF = cls([cls.ViewDir.Up, cls.ViewDir.Left, cls.ViewDir.Forward])
        cls.UFL = cls([cls.ViewDir.Up, cls.ViewDir.Forward, cls.ViewDir.Left])
        cls.LUF = cls([cls.ViewDir.Left, cls.ViewDir.Up, cls.ViewDir.Forward])
        cls.LFU = cls([cls.ViewDir.Left, cls.ViewDir.Forward, cls.ViewDir.Up])
        cls.FUL = cls([cls.ViewDir.Forward, cls.ViewDir.Up, cls.ViewDir.Left])
        cls.FLU = cls([cls.ViewDir.Forward, cls.ViewDir.Left, cls.ViewDir.Up])
        cls.ULB = cls([cls.ViewDir.Up, cls.ViewDir.Left, cls.ViewDir.Back])
        cls.UBL = cls([cls.ViewDir.Up, cls.ViewDir.Back, cls.ViewDir.Left])
        cls.LUB = cls([cls.ViewDir.Left, cls.ViewDir.Up, cls.ViewDir.Back])
        cls.LBU = cls([cls.ViewDir.Left, cls.ViewDir.Back, cls.ViewDir.Up])
        cls.BUL = cls([cls.ViewDir.Back, cls.ViewDir.Up, cls.ViewDir.Left])
        cls.BLU = cls([cls.ViewDir.Back, cls.ViewDir.Left, cls.ViewDir.Up])
        cls.URF = cls([cls.ViewDir.Up, cls.ViewDir.Right, cls.ViewDir.Forward])
        cls.UFR = cls([cls.ViewDir.Up, cls.ViewDir.Forward, cls.ViewDir.Right])
        cls.RUF = cls([cls.ViewDir.Right, cls.ViewDir.Up, cls.ViewDir.Forward])
        cls.RFU = cls([cls.ViewDir.Right, cls.ViewDir.Forward, cls.ViewDir.Up])
        cls.FUR = cls([cls.ViewDir.Forward, cls.ViewDir.Up, cls.ViewDir.Right])
        cls.FRU = cls([cls.ViewDir.Forward, cls.ViewDir.Right, cls.ViewDir.Up])
        cls.URB = cls([cls.ViewDir.Up, cls.ViewDir.Right, cls.ViewDir.Back])
        cls.UBR = cls([cls.ViewDir.Up, cls.ViewDir.Back, cls.ViewDir.Right])
        cls.RUB = cls([cls.ViewDir.Right, cls.ViewDir.Up, cls.ViewDir.Back])
        cls.RBU = cls([cls.ViewDir.Right, cls.ViewDir.Back, cls.ViewDir.Up])
        cls.BUR = cls([cls.ViewDir.Back, cls.ViewDir.Up, cls.ViewDir.Right])
        cls.BRU = cls([cls.ViewDir.Back, cls.ViewDir.Right, cls.ViewDir.Up])
        cls.DLF = cls([cls.ViewDir.Down, cls.ViewDir.Left, cls.ViewDir.Forward])
        cls.DFL = cls([cls.ViewDir.Down, cls.ViewDir.Forward, cls.ViewDir.Left])
        cls.LDF = cls([cls.ViewDir.Left, cls.ViewDir.Down, cls.ViewDir.Forward])
        cls.LFD = cls([cls.ViewDir.Left, cls.ViewDir.Forward, cls.ViewDir.Down])
        cls.FDL = cls([cls.ViewDir.Forward, cls.ViewDir.Down, cls.ViewDir.Left])
        cls.FLD = cls([cls.ViewDir.Forward, cls.ViewDir.Left, cls.ViewDir.Down])
        cls.DLB = cls([cls.ViewDir.Down, cls.ViewDir.Left, cls.ViewDir.Back])
        cls.DBL = cls([cls.ViewDir.Down, cls.ViewDir.Back, cls.ViewDir.Left])
        cls.LDB = cls([cls.ViewDir.Left, cls.ViewDir.Down, cls.ViewDir.Back])
        cls.LBD = cls([cls.ViewDir.Left, cls.ViewDir.Back, cls.ViewDir.Down])
        cls.BDL = cls([cls.ViewDir.Back, cls.ViewDir.Down, cls.ViewDir.Left])
        cls.BLD = cls([cls.ViewDir.Back, cls.ViewDir.Left, cls.ViewDir.Down])
        cls.DRF = cls([cls.ViewDir.Down, cls.ViewDir.Right, cls.ViewDir.Forward])
        cls.DFR = cls([cls.ViewDir.Down, cls.ViewDir.Forward, cls.ViewDir.Right])
        cls.RDF = cls([cls.ViewDir.Right, cls.ViewDir.Down, cls.ViewDir.Forward])
        cls.RFD = cls([cls.ViewDir.Right, cls.ViewDir.Forward, cls.ViewDir.Down])
        cls.FDR = cls([cls.ViewDir.Forward, cls.ViewDir.Down, cls.ViewDir.Right])
        cls.FRD = cls([cls.ViewDir.Forward, cls.ViewDir.Right, cls.ViewDir.Down])
        cls.DRB = cls([cls.ViewDir.Down, cls.ViewDir.Right, cls.ViewDir.Back])
        cls.DBR = cls([cls.ViewDir.Down, cls.ViewDir.Back, cls.ViewDir.Right])
        cls.RDB = cls([cls.ViewDir.Right, cls.ViewDir.Down, cls.ViewDir.Back])
        cls.RBD = cls([cls.ViewDir.Right, cls.ViewDir.Back, cls.ViewDir.Down])
        cls.BDR = cls([cls.ViewDir.Back, cls.ViewDir.Down, cls.ViewDir.Right])
        cls.BRD = cls([cls.ViewDir.Back, cls.ViewDir.Right, cls.ViewDir.Down])
        cls.RIGHT_HAND_X_UP = cls([cls.ViewDir.Up, cls.ViewDir.Right, cls.ViewDir.Forward])
        cls.RIGHT_HAND_X_DOWN = cls([cls.ViewDir.Down, cls.ViewDir.Right, cls.ViewDir.Back])
        cls.RIGHT_HAND_Y_UP = cls([cls.ViewDir.Right, cls.ViewDir.Up, cls.ViewDir.Back])
        cls.RIGHT_HAND_Y_DOWN = cls([cls.ViewDir.Right, cls.ViewDir.Down, cls.ViewDir.Forward])
        cls.RIGHT_HAND_Z_UP = cls([cls.ViewDir.Right, cls.ViewDir.Forward, cls.ViewDir.Up])
        cls.RIGHT_HAND_Z_DOWN = cls([cls.ViewDir.Right, cls.ViewDir.Back, cls.ViewDir.Down])
        cls.LEFT_HAND_X_UP = cls([cls.ViewDir.Up, cls.ViewDir.Right, cls.ViewDir.Back])
        cls.LEFT_HAND_X_DOWN = cls([cls.ViewDir.Down, cls.ViewDir.Right, cls.ViewDir.Forward])
        cls.LEFT_HAND_Y_UP = cls([cls.ViewDir.Right, cls.ViewDir.Up, cls.ViewDir.Forward])
        cls.LEFT_HAND_Y_DOWN = cls([cls.ViewDir.Right, cls.ViewDir.Down, cls.ViewDir.Back])
        cls.LEFT_HAND_Z_UP = cls([cls.ViewDir.Right, cls.ViewDir.Back, cls.ViewDir.Up])
        cls.LEFT_HAND_Z_DOWN = cls([cls.ViewDir.Right, cls.ViewDir.Forward, cls.ViewDir.Down])