from __future__ import annotations
from typing import Any
import numpy as np
import pytest
import rerun as rr
import torch
from rerun.datatypes import TensorBuffer, TensorBufferType, TensorData, TensorDataLike, TensorDimension
rng = np.random.default_rng(12345)
RANDOM_IMAGE_SOURCE = rng.integers(0, 255, size=(10, 20))
IMAGE_INPUTS: list[TensorDataLike] = [TensorData(shape=[TensorDimension(10, 'height'), TensorDimension(20, 'width')], buffer=TensorBuffer(RANDOM_IMAGE_SOURCE)), RANDOM_IMAGE_SOURCE]

def segmentation_image_image_expected() -> Any:
    if False:
        i = 10
        return i + 15
    return rr.SegmentationImage(data=RANDOM_IMAGE_SOURCE)

def test_image() -> None:
    if False:
        for i in range(10):
            print('nop')
    expected = segmentation_image_image_expected()
    for img in IMAGE_INPUTS:
        arch = rr.SegmentationImage(data=img)
        assert arch == expected
GOOD_IMAGE_INPUTS: list[TensorDataLike] = [rng.integers(0, 255, (10, 20)), rng.integers(0, 255, (1, 10, 20)), rng.integers(0, 255, (10, 20, 1)), torch.randint(0, 255, (10, 20))]
BAD_IMAGE_INPUTS: list[TensorDataLike] = [rng.integers(0, 255, (10, 20, 3)), rng.integers(0, 255, (10, 20, 4)), rng.integers(0, 255, (10,)), rng.integers(0, 255, (1, 10, 20, 3)), rng.integers(0, 255, (1, 10, 20, 4)), rng.integers(0, 255, (10, 20, 3, 1)), rng.integers(0, 255, (10, 20, 4, 1)), rng.integers(0, 255, (10, 20, 2)), rng.integers(0, 255, (10, 20, 5)), rng.integers(0, 255, (10, 20, 3, 2))]

def test_segmentation_image_shapes() -> None:
    if False:
        while True:
            i = 10
    import rerun as rr
    rr.set_strict_mode(True)
    for img in GOOD_IMAGE_INPUTS:
        rr.DepthImage(img)
    for img in BAD_IMAGE_INPUTS:
        with pytest.raises(ValueError):
            rr.DepthImage(img)

def test_segmentation_coercion() -> None:
    if False:
        for i in range(10):
            print('nop')
    seg_img = np.require(RANDOM_IMAGE_SOURCE, np.float32)
    seg = rr.SegmentationImage(seg_img)
    U16_TYPE_ID = list((f.name for f in TensorBufferType().storage_type)).index('U16')
    assert seg.data.as_arrow_array().storage.field(1)[0].type_code == U16_TYPE_ID