import jittor as jt
import numpy as np
import unittest
try:
    import torch
    from torch.autograd import Variable
    has_autograd = True
except:
    has_autograd = False

@unittest.skipIf(not has_autograd or not jt.compiler.has_cuda, 'No autograd or cuda found.')
class TestDigamma(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        jt.flags.use_cuda = 1

    def tearDown(self):
        if False:
            while True:
                i = 10
        jt.flags.use_cuda = 0

    def test_digamma(self):
        if False:
            while True:
                i = 10
        for i in range(30):
            concentration = np.random.uniform(1, 3)
            rate = np.random.uniform(1, 2)
            j_gamma = jt.distributions.GammaDistribution(concentration, rate)
            t_gamma = torch.distributions.gamma.Gamma(torch.tensor([concentration]), torch.tensor([rate]))
            samples = t_gamma.sample((30, i + 5))
            j_samples = jt.array(samples.detach().numpy())
            np.testing.assert_allclose(j_gamma.log_prob(j_samples).data, t_gamma.log_prob(samples).detach().numpy(), rtol=0.0001, atol=1e-06)
            samples = j_gamma.sample((30, i + 5))
            t_samples = torch.tensor(samples.numpy())
            np.testing.assert_allclose(j_gamma.log_prob(samples).data, t_gamma.log_prob(t_samples).detach().numpy(), rtol=0.0001, atol=1e-06)
if __name__ == '__main__':
    unittest.main()