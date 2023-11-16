#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import pytest
import os
from unittest import TestCase
from bigdl.nano.pytorch.vision.models import vision
from test.pytorch.utils._train_torch_lightning import train_with_linear_top_layer


batch_size = 256
num_workers = 0
data_dir = "/tmp/data"


class TestVisionTransforms(TestCase):

    def test_if_transforms_missing(self):
        from bigdl.nano.pytorch.vision.transforms import transforms as nano_module
        from torchvision.transforms import transforms as compared_module

        assert all([x in nano_module.__all__ for x in compared_module.__all__]), \
            "Missing transforms"


if __name__ == '__main__':
    pytest.main([__file__])
