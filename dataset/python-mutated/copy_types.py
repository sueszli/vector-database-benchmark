from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import Self
import cudf
import cudf._lib as libcudf
from cudf._lib.types import size_type_dtype
if TYPE_CHECKING:
    from cudf.core.column import NumericalColumn

@dataclass
class GatherMap:
    """A representation of a column as a gather map.

    This object augments the column with the information that it
    is valid as a gather map for the specified number of rows with
    the given nullification flag.

    Parameters
    ----------
    column
        The data to turn into a column and then verify
    nrows
        The number of rows to verify against
    nullify
        Will the gather map be used nullifying out of bounds
        accesses?

    Returns
    -------
    GatherMap
        New object wrapping the column bearing witness to its
        suitability as a gather map for columns with nrows.

    Raises
    ------
    TypeError
        If the column is of unsuitable dtype
    IndexError
        If the map is not in bounds.
    """
    column: 'NumericalColumn'
    nrows: int
    nullify: bool

    def __init__(self, column: Any, nrows: int, *, nullify: bool):
        if False:
            i = 10
            return i + 15
        self.column = cudf.core.column.as_column(column)
        self.nrows = nrows
        self.nullify = nullify
        if len(self.column) == 0:
            self.column = cast('NumericalColumn', self.column.astype(size_type_dtype))
        else:
            if self.column.dtype.kind not in {'i', 'u'}:
                raise TypeError('Gather map must have integer dtype')
            if not nullify:
                (lo, hi) = libcudf.reduce.minmax(self.column)
                if lo.value < -nrows or hi.value >= nrows:
                    raise IndexError(f'Gather map is out of bounds for [0, {nrows})')

    @classmethod
    def from_column_unchecked(cls, column: 'NumericalColumn', nrows: int, *, nullify: bool) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Construct a new GatherMap from a column without checks.\n\n        Parameters\n        ----------\n        column\n           The column that will be used as a gather map\n        nrows\n           The number of rows the gather map will be used for\n        nullify\n           Will the gather map be used nullifying out of bounds\n           accesses?\n\n        Returns\n        -------\n        GatherMap\n\n        Notes\n        -----\n        This method asserts, by fiat, that the column is valid.\n        Behaviour is undefined if it is not.\n        '
        self = cls.__new__(cls)
        self.column = column
        self.nrows = nrows
        self.nullify = nullify
        return self

@dataclass
class BooleanMask:
    """A representation of a column as a boolean mask.

    This augments the column with information that it is valid as a
    boolean mask for columns with a given number of rows

    Parameters
    ----------
    column
        The data to turn into a column to then verify
    nrows
        the number of rows to verify against

    Returns
    -------
    BooleanMask
        New object wrapping the column bearing witness to its
        suitability as a boolean mask for columns with matching
        row count.

    Raises
    ------
    TypeError
        If the column is of unsuitable dtype
    IndexError
        If the mask has the wrong number of rows
    """
    column: 'NumericalColumn'

    def __init__(self, column: Any, nrows: int):
        if False:
            print('Hello World!')
        self.column = cudf.core.column.as_column(column)
        if self.column.dtype.kind != 'b':
            raise TypeError('Boolean mask must have bool dtype')
        if len(column) != nrows:
            raise IndexError(f'Column with {len(column)} rows not suitable as a boolean mask for {nrows} rows')

    @classmethod
    def from_column_unchecked(cls, column: 'NumericalColumn') -> Self:
        if False:
            while True:
                i = 10
        'Construct a new BooleanMask from a column without checks.\n\n        Parameters\n        ----------\n        column\n           The column that will be used as a boolean mask\n\n        Returns\n        -------\n        BooleanMask\n\n        Notes\n        -----\n        This method asserts, by fiat, that the column is valid.\n        Behaviour is undefined if it is not.\n        '
        self = cls.__new__(cls)
        self.column = column
        return self