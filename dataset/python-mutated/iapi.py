"""
Internal API

Specifications for containers that will hold different kinds
of objects with data created when the plot is being built.
"""
from __future__ import annotations
import typing
from copy import copy
from dataclasses import dataclass, fields
if typing.TYPE_CHECKING:
    from typing import Any, Dict, Iterator, Optional, Sequence
    from plotnine.typing import Axes, CoordRange, Figure, FloatArrayLike, Scale, ScaleBreaks, ScaledAestheticsName, ScaleLimits, StripPosition, TupleFloat2

@dataclass
class scale_view:
    """
    Scale information after it has been trained
    """
    scale: Scale
    aesthetics: list[ScaledAestheticsName]
    name: Optional[str]
    limits: ScaleLimits
    range: CoordRange
    breaks: ScaleBreaks
    minor_breaks: FloatArrayLike
    labels: Sequence[str]

@dataclass
class range_view:
    """
    Range information after trainning
    """
    range: TupleFloat2
    range_coord: TupleFloat2

@dataclass
class labels_view:
    """
    Scale labels (incl. caption & title) for the plot
    """
    x: Optional[str] = None
    y: Optional[str] = None
    alpha: Optional[str] = None
    color: Optional[str] = None
    colour: Optional[str] = None
    fill: Optional[str] = None
    linetype: Optional[str] = None
    shape: Optional[str] = None
    size: Optional[str] = None
    stroke: Optional[str] = None
    title: Optional[str] = None
    caption: Optional[str] = None
    subtitle: Optional[str] = None

    def update(self, other: labels_view):
        if False:
            return 10
        '\n        Update the labels with those in other\n        '
        for (name, value) in other.iter_set_fields():
            setattr(self, name, value)

    def add_defaults(self, other: labels_view):
        if False:
            return 10
        '\n        Update labels that are missing with those in other\n        '
        for (name, value) in other.iter_set_fields():
            cur_value = getattr(self, name)
            if cur_value is None:
                setattr(self, name, value)

    def iterfields(self) -> Iterator[tuple[str, Optional[str]]]:
        if False:
            i = 10
            return i + 15
        '\n        Return an iterator of all (field, value) pairs\n        '
        return ((f.name, getattr(self, f.name)) for f in fields(self))

    def iter_set_fields(self) -> Iterator[tuple[str, str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return an iterator of (field, value) pairs of none None values\n        '
        return ((k, v) for (k, v) in self.iterfields() if v is not None)

    def get(self, name: str, default: str) -> str:
        if False:
            return 10
        '\n        Get label value, return default if value is None\n        '
        value = getattr(self, name)
        return str(value) if value is not None else default

    def __contains__(self, name: str) -> bool:
        if False:
            while True:
                i = 10
        '\n        Return True if name has been set (is not None)\n        '
        return getattr(self, name) is not None

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        '\n        Representations without the None values\n        '
        nv_pairs = ', '.join((f'{name}={repr(value)}' for (name, value) in self.iter_set_fields()))
        return f'{self.__class__.__name__}({nv_pairs})'

@dataclass
class panel_view:
    """
    Information from the trained position scales in a panel
    """
    x: scale_view
    y: scale_view

@dataclass
class panel_ranges:
    """
    Ranges for the panel
    """
    x: TupleFloat2
    y: TupleFloat2

@dataclass
class pos_scales:
    """
    Position Scales
    """
    x: Scale
    y: Scale

@dataclass
class mpl_save_view:
    """
    Everything required to save a matplotlib figure
    """
    figure: Figure
    kwargs: Dict[str, Any]

@dataclass
class layout_details:
    """
    Layout information
    """
    panel_index: int
    panel: int
    row: int
    col: int
    scale_x: int
    scale_y: int
    axis_x: bool
    axis_y: bool
    variables: dict[str, Any]

@dataclass
class strip_draw_info:
    """
    Information required to draw strips
    """
    x: float
    y: float
    ha: str
    va: str
    box_width: float
    box_height: float
    strip_text_margin: float
    strip_align: float
    position: StripPosition
    label: str
    ax: Axes
    rotation: float

@dataclass
class strip_label_details:
    """
    Strip Label Details
    """
    variables: dict[str, str]
    meta: dict[str, Any]

    @staticmethod
    def make(layout_info: layout_details, vars: list[str], location: StripPosition) -> strip_label_details:
        if False:
            print('Hello World!')
        variables: dict[str, Any] = {v: str(layout_info.variables[v]) for v in vars}
        meta: dict[str, Any] = {'dimension': 'cols' if location == 'top' else 'rows'}
        return strip_label_details(variables, meta)

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Number of variables\n        '
        return len(self.variables)

    def __copy__(self) -> strip_label_details:
        if False:
            for i in range(10):
                print('nop')
        '\n        Make a copy\n        '
        return strip_label_details(self.variables.copy(), self.meta.copy())

    def copy(self) -> strip_label_details:
        if False:
            while True:
                i = 10
        '\n        Make a copy\n        '
        return copy(self)

    def text(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Strip text\n\n        Join the labels for all the variables along a\n        dimension\n        '
        return '\n'.join(list(self.variables.values()))

    def collapse(self) -> strip_label_details:
        if False:
            i = 10
            return i + 15
        '\n        Concatenate all label values into one item\n        '
        result = self.copy()
        result.variables = {'value': ', '.join(result.variables.values())}
        return result