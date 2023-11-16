"""
Test that the fiftyone core package does not depend on any extra packages that
are intended to be manually installed by users.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import sys
import pytest
sys.modules['tensorflow'] = None
sys.modules['tensorflow_datasets'] = None
sys.modules['torch'] = None
sys.modules['torchvision'] = None
sys.modules['flash'] = None
sys.modules['pycocotools'] = None

def test_import_core():
    if False:
        while True:
            i = 10
    import fiftyone