from __future__ import annotations
from manimlib.animation.transform import Transform
from manimlib.constants import PI
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    from manimlib.mobject.geometry import Arrow
    from manimlib.mobject.mobject import Mobject
    from manimlib.typing import ManimColor

class GrowFromPoint(Transform):

    def __init__(self, mobject: Mobject, point: np.ndarray, point_color: ManimColor=None, **kwargs):
        if False:
            i = 10
            return i + 15
        self.point = point
        self.point_color = point_color
        super().__init__(mobject, **kwargs)

    def create_target(self) -> Mobject:
        if False:
            i = 10
            return i + 15
        return self.mobject.copy()

    def create_starting_mobject(self) -> Mobject:
        if False:
            while True:
                i = 10
        start = super().create_starting_mobject()
        start.scale(0)
        start.move_to(self.point)
        if self.point_color is not None:
            start.set_color(self.point_color)
        return start

class GrowFromCenter(GrowFromPoint):

    def __init__(self, mobject: Mobject, **kwargs):
        if False:
            return 10
        point = mobject.get_center()
        super().__init__(mobject, point, **kwargs)

class GrowFromEdge(GrowFromPoint):

    def __init__(self, mobject: Mobject, edge: np.ndarray, **kwargs):
        if False:
            print('Hello World!')
        point = mobject.get_bounding_box_point(edge)
        super().__init__(mobject, point, **kwargs)

class GrowArrow(GrowFromPoint):

    def __init__(self, arrow: Arrow, **kwargs):
        if False:
            while True:
                i = 10
        point = arrow.get_start()
        super().__init__(arrow, point, **kwargs)