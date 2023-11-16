from __future__ import annotations
from typing import Any
import rerun.components as rrc
from rerun.archetypes import ViewCoordinates
from .common_arrays import none_empty_or_value

def view_coordinates_expected(obj: Any) -> rrc.ViewCoordinatesBatch:
    if False:
        while True:
            i = 10
    expected = none_empty_or_value(obj, [rrc.ViewCoordinates.ViewDir.Right, rrc.ViewCoordinates.ViewDir.Down, rrc.ViewCoordinates.ViewDir.Forward])
    return rrc.ViewCoordinatesBatch(expected)
VIEW_COORDINATES_INPUTS: list[rrc.ViewCoordinatesArrayLike | None] = [None, rrc.ViewCoordinates([rrc.ViewCoordinates.ViewDir.Right, rrc.ViewCoordinates.ViewDir.Down, rrc.ViewCoordinates.ViewDir.Forward]), [rrc.ViewCoordinates.ViewDir.Right, rrc.ViewCoordinates.ViewDir.Down, rrc.ViewCoordinates.ViewDir.Forward], rrc.ViewCoordinates.RDF, [rrc.ViewCoordinates.RDF]]

def test_view_coordinates() -> None:
    if False:
        return 10
    for coordinates in VIEW_COORDINATES_INPUTS:
        arch = ViewCoordinates(coordinates)
        print(f'rr.ViewCoordinates(\n    {str(coordinates)}\n)')
        print(f'{arch}\n')
        assert arch.xyz == view_coordinates_expected(coordinates)