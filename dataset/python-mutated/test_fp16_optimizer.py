import copy
import logging
import unittest
import torch
from fairseq.optim.fp16_optimizer import FP16Optimizer, MemoryEfficientFP16Optimizer
from omegaconf import OmegaConf

@unittest.skipIf(not torch.cuda.is_available(), 'test requires a GPU')
class TestGradientScaling(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.x = torch.tensor([2.0]).cuda().half()
        weight = 3.0
        bias = 5.0
        self.error = 1.0
        self.target = torch.tensor([self.x * weight + bias + self.error]).cuda().half()
        self.loss_fn = torch.nn.L1Loss()
        self.model = torch.nn.Linear(1, 1)
        self.model.weight.data = torch.tensor([[weight]])
        self.model.bias.data = torch.tensor([bias])
        self.model.cuda().half()
        self.params = list(self.model.parameters())
        self.cfg_dls = OmegaConf.create({'optimization': {'lr': [0.1]}, 'optimizer': {'_name': 'adam', 'lr': [0.1], 'adam_betas': '(0.9, 0.999)', 'adam_eps': 1e-08, 'weight_decay': 0.0}, 'common': {'fp16_init_scale': 1, 'fp16_scale_window': 1, 'fp16_scale_tolerance': 1, 'threshold_loss_scale': 1, 'min_loss_scale': 0.0001, 'tpu': False}})
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        logging.disable(logging.NOTSET)

    def run_iter(self, model, params, optimizer):
        if False:
            while True:
                i = 10
        optimizer.zero_grad()
        y = model(self.x)
        loss = self.loss_fn(y, self.target)
        optimizer.backward(loss)
        self.assertEqual(loss, torch.tensor(1.0, device='cuda:0', dtype=torch.float16))
        grad_norm = optimizer.clip_grad_norm(0)
        self.assertAlmostEqual(grad_norm.item(), 2.2361, 4)
        optimizer.step()
        self.assertEqual(model.weight, torch.tensor([[3.0996]], device='cuda:0', dtype=torch.float16, requires_grad=True))
        self.assertEqual(model.bias, torch.tensor([5.1016], device='cuda:0', dtype=torch.float16, requires_grad=True))
        self.assertEqual(optimizer.scaler.loss_scale, 2.0)

    def test_mixed_precision(self):
        if False:
            i = 10
            return i + 15
        model = copy.deepcopy(self.model)
        params = list(model.parameters())
        optimizer = FP16Optimizer.build_optimizer(self.cfg_dls, params)
        self.run_iter(model, params, optimizer)
        self.assertTrue(all((torch.all(fp32_params.eq(torch.tensor([3.1, 5.1], device='cuda:0', requires_grad=True))) for fp32_params in optimizer.fp32_params.values())))

    def test_memory_efficient(self):
        if False:
            print('Hello World!')
        model = copy.deepcopy(self.model)
        params = list(model.parameters())
        optimizer = MemoryEfficientFP16Optimizer.build_optimizer(self.cfg_dls, params)
        self.run_iter(model, params, optimizer)
if __name__ == '__main__':
    unittest.main()