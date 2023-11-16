import unittest
from copy import deepcopy
from dataclasses import dataclass
import pytest
from typing import Optional
from unittest.mock import patch
import torch
from fairseq.models.ema import EMA

class DummyModule(torch.nn.Module):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        'LightningModule for testing purposes\n\n        Args:\n            epoch_min_loss_override (int, optional): Pass in an epoch that will be set to the minimum\n                validation loss for testing purposes (zero based). If None this is ignored. Defaults to None.\n        '
        super().__init__()
        self.layer = torch.nn.Linear(in_features=32, out_features=2)
        self.another_layer = torch.nn.Linear(in_features=2, out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        x = self.layer(x)
        return self.another_layer(x)

@dataclass
class EMAConfig(object):
    ema_decay: float = 0.99
    ema_start_update: int = 0
    ema_fp32: bool = False
    ema_seed_model: Optional[str] = None
    ema_update_freq: int = 1

class TestEMA(unittest.TestCase):

    def assertTorchAllClose(self, x, y, atol=1e-08, rtol=1e-05, msg=None):
        if False:
            while True:
                i = 10
        diff = x.float() - y.float()
        diff_norm = torch.norm(diff)
        other_norm = torch.norm(y.float())
        if msg is None:
            msg = '|input - other| > {} + {} * |other|'.format(atol, rtol)
        self.assertLessEqual(diff_norm, atol + rtol * other_norm, msg=msg)

    def test_ema(self):
        if False:
            for i in range(10):
                print('nop')
        model = DummyModule()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        state = deepcopy(model.state_dict())
        config = EMAConfig()
        ema = EMA(model, config)
        ema._set_decay(config.ema_decay)
        self.assertEqual(ema.get_decay(), config.ema_decay)
        self.assertEqual(ema.get_model(), ema.model)
        self.assertEqual(len(ema.fp32_params), 0)
        x = torch.randn(32)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        ema.step(model)
        ema_state_dict = ema.get_model().state_dict()
        for (key, param) in model.state_dict().items():
            prev_param = state[key]
            ema_param = ema_state_dict[key]
            if 'version' in key:
                continue
            self.assertTorchAllClose(ema_param, config.ema_decay * prev_param + (1 - config.ema_decay) * param)
        self.assertEqual(len(ema.fp32_params), 0)
        model2 = DummyModule()
        ema.reverse(model2)
        for (key, param) in model2.state_dict().items():
            ema_param = ema_state_dict[key]
            self.assertTrue(torch.allclose(ema_param, param))
        with patch.object(ema, '_step_internal', return_value=None) as mock_method:
            ema.step(model)
            mock_method.assert_called_once_with(model, None)

    def _test_ema_start_update(self, updates):
        if False:
            i = 10
            return i + 15
        model = DummyModule()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        state = deepcopy(model.state_dict())
        config = EMAConfig(ema_start_update=1)
        ema = EMA(model, config)
        x = torch.randn(32)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        ema.step(model, updates=updates)
        ema_state_dict = ema.get_model().state_dict()
        self.assertEqual(ema.get_decay(), 0 if updates == 0 else config.ema_decay)
        for (key, param) in model.state_dict().items():
            ema_param = ema_state_dict[key]
            prev_param = state[key]
            if 'version' in key:
                continue
            if updates == 0:
                self.assertTorchAllClose(ema_param, param)
            else:
                self.assertTorchAllClose(ema_param, config.ema_decay * prev_param + (1 - config.ema_decay) * param)
        with patch.object(ema, '_step_internal', return_value=None) as mock_method:
            ema.step(model, updates=updates)
            mock_method.assert_called_once_with(model, updates)

    def test_ema_before_start_update(self):
        if False:
            while True:
                i = 10
        self._test_ema_start_update(updates=0)

    def test_ema_after_start_update(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_ema_start_update(updates=1)

    def test_ema_fp32(self):
        if False:
            while True:
                i = 10
        dtype = torch.float
        model = DummyModule().to(dtype)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        state = deepcopy(model.state_dict())
        config = EMAConfig(ema_fp32=True)
        ema = EMA(model, config)
        x = torch.randn(32)
        y = model(x.to(dtype))
        loss = y.sum()
        loss.backward()
        optimizer.step()
        ema.step(model)
        for (key, param) in model.state_dict().items():
            prev_param = state[key]
            ema_param = ema.get_model().state_dict()[key]
            if 'version' in key:
                continue
            self.assertIn(key, ema.fp32_params)
            self.assertLessEqual(torch.norm(ema_param.float() - (config.ema_decay * prev_param.float() + (1 - config.ema_decay) * param.float()).to(dtype).float()), torch.norm(ema_param.float() - (config.ema_decay * prev_param + (1 - config.ema_decay) * param).float()))
            self.assertTorchAllClose(ema_param, (config.ema_decay * prev_param.float() + (1 - config.ema_decay) * param.float()).to(dtype))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CPU no longer supports Linear in half precision')
    def test_ema_fp16(self):
        if False:
            i = 10
            return i + 15
        model = DummyModule().cuda().half()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        state = deepcopy(model.state_dict())
        config = EMAConfig(ema_fp32=False)
        ema = EMA(model, config)
        self.assertEqual(len(ema.fp32_params), 0)
        x = torch.randn(32).cuda()
        y = model(x.half())
        loss = y.sum()
        loss.backward()
        optimizer.step()
        ema.step(model)
        for (key, param) in model.state_dict().items():
            prev_param = state[key]
            ema_param = ema.get_model().state_dict()[key]
            if 'version' in key:
                continue
            self.assertLessEqual(torch.norm(ema_param.float() - (config.ema_decay * prev_param + (1 - config.ema_decay) * param).float()), torch.norm(ema_param.float() - (config.ema_decay * prev_param.float() + (1 - config.ema_decay) * param.float()).half().float()))
            self.assertTorchAllClose(ema_param, config.ema_decay * prev_param + (1 - config.ema_decay) * param)
        self.assertEqual(len(ema.fp32_params), 0)
if __name__ == '__main__':
    unittest.main()