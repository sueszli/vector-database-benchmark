import unittest
import jittor as jt
import numpy as np
import jittor.models as jtmodels

def load_parameters(m1, m2):
    if False:
        for i in range(10):
            print('nop')
    m1.save('/tmp/temp.pk')
    m2.load('/tmp/temp.pk')

def compare_parameters(m1, m2):
    if False:
        for i in range(10):
            print('nop')
    ps1 = m1.parameters()
    ps2 = m2.parameters()
    for i in range(len(ps1)):
        x = ps1[i].data + 1e-08
        y = ps2[i].data + 1e-08
        relative_error = abs(x - y) / abs(y)
        diff = relative_error.mean()
        assert diff < 0.0001, (diff, 'backward', ps2[i].name(), ps1[i].mean(), ps1[i].std(), ps2[i].mean(), ps2[i].std())

class TestDepthwiseConv(unittest.TestCase):

    @unittest.skipIf(not jt.has_cuda, 'Cuda not found')
    @jt.flag_scope(use_cuda=1)
    def test_data(self):
        if False:
            while True:
                i = 10
        test_img = np.random.random((64, 3, 224, 224)).astype('float32')
        jittor_test_img = jt.array(test_img)
        lr = 100
        jittor_model = jtmodels.__dict__['mobilenet_v2']()
        jittor_model2 = jtmodels.__dict__['mobilenet_v2']()
        jittor_model.train()
        jittor_model.classifier[0].eval()
        for m in jittor_model.modules():
            if isinstance(m, jt.nn.BatchNorm):
                m.eval()
        jittor_model2.train()
        jittor_model2.classifier[0].eval()
        for m in jittor_model2.modules():
            if isinstance(m, jt.nn.BatchNorm):
                m.eval()
        load_parameters(jittor_model2, jittor_model)
        for m in jittor_model.modules():
            if isinstance(m, jt.nn.Conv):
                m.is_depthwise_conv = False
        cnt = 0
        for m in jittor_model2.modules():
            if isinstance(m, jt.nn.Conv):
                if m.is_depthwise_conv:
                    cnt += 1
        assert cnt == 17, (cnt, '!=', 17)
        jt_optimizer = jt.nn.SGD(jittor_model.parameters(), lr=lr)
        jt_optimizer2 = jt.nn.SGD(jittor_model2.parameters(), lr=lr)
        jittor_result = jittor_model(jittor_test_img)
        mask = jt.random(jittor_result.shape, jittor_result.dtype)
        loss = jittor_result * mask
        jt_optimizer.step(loss)
        jt.sync_all(True)
        jittor_result2 = jittor_model2(jittor_test_img)
        loss = jittor_result2 * mask
        x = jittor_result2.data + 1e-08
        y = jittor_result.data + 1e-08
        relative_error = abs(x - y) / abs(y)
        diff = relative_error.mean()
        assert diff < 0.0001, (diff, 'forword')
        jt_optimizer2.step(loss)
        jt.sync_all(True)
        compare_parameters(jittor_model, jittor_model2)
        jt.clean()
        jt.gc()
if __name__ == '__main__':
    unittest.main()