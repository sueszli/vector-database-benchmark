import unittest
import numpy as np
import os
import sys
import jittor as jt
skip_this_test = False
try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    import torchvision.models as tcmodels
    from torch import nn
except:
    torch = None
    skip_this_test = True

@unittest.skipIf(skip_this_test, 'skip_this_test')
class TestAutoDiff(unittest.TestCase):

    def test_pt_hook(self):
        if False:
            for i in range(10):
                print('nop')
        code = '\nimport numpy as np\nfrom jittor_utils import auto_diff\nimport torch\nimport torchvision.models as tcmodels\nnet = tcmodels.resnet50()\nnet.train()\nhook = auto_diff.Hook("resnet50")\nhook.hook_module(net)\n\nnp.random.seed(0)\ndata = np.random.random((2,3,224,224)).astype(\'float32\')\ndata = torch.Tensor(data)\nnet(data)\n# assert auto_diff.has_error == 0, auto_diff.has_error\n'
        with open('/tmp/test_pt_hook.py', 'w') as f:
            f.write(code)
        print(jt.flags.cache_path)
        os.system(f'rm -rf {jt.flags.cache_path}/../../auto_diff/resnet50')
        assert os.system(sys.executable + ' /tmp/test_pt_hook.py') == 0
        assert os.system(sys.executable + ' /tmp/test_pt_hook.py') == 0
        code = '\nimport numpy as np\nimport jittor as jt\nfrom jittor_utils import auto_diff\nfrom jittor.models import resnet50\nnet = resnet50()\nnet.train()\nhook = auto_diff.Hook("resnet50")\nhook.hook_module(net)\n\nnp.random.seed(0)\ndata = np.random.random((2,3,224,224)).astype(\'float32\')\ndata = jt.array(data)\nnet(data)\n# assert auto_diff.has_error == 0, auto_diff.has_error\n'
        with open('/tmp/test_jt_hook.py', 'w') as f:
            f.write(code)
        assert os.system(sys.executable + ' /tmp/test_jt_hook.py') == 0
if __name__ == '__main__':
    unittest.main()