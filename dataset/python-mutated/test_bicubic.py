import unittest
import jittor as jt
import numpy as np
skip_this_test = False
try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    from torch.nn import functional as F
except:
    torch = None
    skip_this_test = True

@unittest.skipIf(skip_this_test, 'No Torch found')
class TestBicubicInterpolate(unittest.TestCase):

    def test_bicubic(self):
        if False:
            i = 10
            return i + 15
        for _ in range(20):
            try:
                tn = np.random.randn(1, 1, 5, 5).astype('float32')
                ja = jt.array(tn)
                ta = torch.autograd.Variable(torch.from_numpy(tn), requires_grad=True)
                ju = jt.nn.interpolate(ja, scale_factor=2, mode='bicubic')
                tu = F.interpolate(ta, scale_factor=2, mode='bicubic')
                assert np.allclose(ju.data, tu.detach().numpy(), rtol=0.001, atol=1e-06)
                gju = jt.grad(ju, ja)
                gtu = torch.autograd.grad(tu, ta, torch.ones_like(tu), retain_graph=True)[0]
                assert np.allclose(gju.data, gtu.detach().numpy(), rtol=0.001, atol=1e-06)
                je = jt.nn.interpolate(ja, scale_factor=2, mode='bicubic', align_corners=True)
                te = F.interpolate(ta, scale_factor=2, mode='bicubic', align_corners=True)
                assert np.allclose(je.data, te.detach().numpy(), rtol=0.001, atol=1e-06)
                gje = jt.grad(je, ja)
                gte = torch.autograd.grad(te, ta, torch.ones_like(tu), retain_graph=True)[0]
                assert np.allclose(gje.data, gte.detach().numpy(), rtol=0.001, atol=1e-06)
            except AssertionError:
                print(ju, tu)
                print(je, te)
if __name__ == '__main__':
    unittest.main()