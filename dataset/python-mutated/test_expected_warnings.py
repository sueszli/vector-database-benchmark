from __future__ import annotations
import pytest
import rerun as rr
from rerun.error_utils import RerunWarning
rr.init('rerun_example_exceptions', spawn=False)
mem = rr.memory_recording()

def test_expected_warnings() -> None:
    if False:
        while True:
            i = 10
    rr.set_strict_mode(False)
    with pytest.warns(RerunWarning) as warnings:
        expected_warnings = [(rr.log('points', rr.Points3D([1, 2, 3, 4, 5])), 'Expected either a flat array with a length multiple of 3 elements, or an array with shape (`num_elements`, 3). Shape of passed array was (5,).'), (rr.log('points', rr.Points2D([1, 2, 3, 4, 5])), 'Expected either a flat array with a length multiple of 2 elements, or an array with shape (`num_elements`, 2). Shape of passed array was (5,).'), (rr.log('test_transform', rr.Transform3D(translation=[1, 2, 3, 4])), 'translation must be compatible with Vec3D'), (rr.log('test_transform', rr.Transform3D(rotation=[1, 2, 3, 4, 5])), 'rotation must be compatible with Rotation3D'), (rr.log('test_transform', rr.Transform3D(mat3x3=[1, 2, 3, 4, 5])), 'cannot reshape array of size 5 into shape (3,3))'), (rr.log('test_transform', rr.TranslationAndMat3x3(translation=[1, 0, 0])), 'Expected an object implementing rerun.AsComponents or an iterable of rerun.ComponentBatchLike, but got'), (rr.log('world/image', rr.Pinhole(focal_length=3)), 'Must provide one of principal_point, resolution, or width/height)')]
        assert len(warnings) == len(expected_warnings)
        for (warning, (_, expected)) in zip(warnings, expected_warnings):
            assert expected in str(warning)