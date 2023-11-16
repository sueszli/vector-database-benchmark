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

@unittest.skipIf(skip_this_test, 'No Torch Found')
class TestFoldOp(unittest.TestCase):

    def test_fold(self):
        if False:
            return 10
        for i in range(4, 10):
            tn = np.random.randn(1, 3, i, i).astype('float32')
            ja = jt.array(tn)
            ta = torch.autograd.Variable(torch.from_numpy(tn), requires_grad=True)
            juf = jt.nn.unfold(ja, kernel_size=2, stride=2, dilation=2, padding=2)
            tuf = F.unfold(ta, kernel_size=2, stride=2, dilation=2, padding=2)
            assert np.allclose(juf.data, tuf.detach().numpy())
            gjuf = jt.grad(juf, ja)
            gtuf = torch.autograd.grad(tuf, ta, torch.ones_like(tuf), retain_graph=True)[0]
            assert np.allclose(gjuf.data, gtuf.detach().numpy())
            jf = jt.nn.fold(juf, output_size=(i, i), kernel_size=2, stride=2, dilation=2, padding=2)
            tf = F.fold(tuf, output_size=(i, i), kernel_size=2, stride=2, dilation=2, padding=2)
            assert np.allclose(jf.data, tf.detach().numpy())
            gjf = jt.grad(jf, juf)
            gtf = torch.autograd.grad(tf, tuf, torch.ones_like(tf), retain_graph=True)[0]
            assert np.allclose(gjf.data, gtf.detach().numpy())
if __name__ == '__main__':
    unittest.main()