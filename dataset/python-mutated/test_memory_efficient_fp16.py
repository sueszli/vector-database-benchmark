import argparse
import logging
import unittest
import torch
from fairseq.optim.adam import FairseqAdam
from fairseq.optim.fp16_optimizer import MemoryEfficientFP16Optimizer
from omegaconf import OmegaConf

@unittest.skipIf(not torch.cuda.is_available(), 'test requires a GPU')
class TestMemoryEfficientFP16(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        logging.disable(logging.NOTSET)

    def test_load_state_dict(self):
        if False:
            while True:
                i = 10
        model = torch.nn.Linear(5, 5).cuda().half()
        params = list(model.parameters())
        optimizer = FairseqAdam(cfg=OmegaConf.create(vars(argparse.Namespace(adam_betas='(0.9, 0.999)', adam_eps=1e-08, weight_decay=0.0, lr=[1e-05]))), params=params)
        me_optimizer = MemoryEfficientFP16Optimizer(cfg=OmegaConf.create({'common': vars(argparse.Namespace(fp16_init_scale=1, fp16_scale_window=1, fp16_scale_tolerance=1, threshold_loss_scale=1, min_loss_scale=0.0001))}), params=params, optimizer=optimizer)
        loss = model(torch.rand(5).cuda().half()).sum()
        me_optimizer.backward(loss)
        me_optimizer.step()
        state = me_optimizer.state_dict()
        me_optimizer.load_state_dict(state)
        for (k, v) in me_optimizer.optimizer.state.items():
            self.assertTrue(k.dtype == torch.float16)
            for v_i in v.values():
                if torch.is_tensor(v_i):
                    self.assertTrue(v_i.dtype == torch.float32)
if __name__ == '__main__':
    unittest.main()