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

from bigdl.chronos.utils import LazyImport
torch = LazyImport('torch')
LinexLoss = LazyImport('bigdl.chronos.pytorch.loss.LinexLoss')

from unittest import TestCase
import pytest

from ... import op_torch


@op_torch
class TestChronosPytorchLoss(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_linex_loss(self):
        y = torch.rand(100, 10, 2)
        yhat_high = y + 1
        yhat_low = y - 1

        # when a is set to > 0, this loss panelize underestimate more.
        loss = LinexLoss(1)
        assert loss(yhat_high, y) < loss(yhat_low, y)

        # when a is set to < 0, this loss panelize overestimate more.
        loss = LinexLoss(-1)
        assert loss(yhat_high, y) > loss(yhat_low, y)
