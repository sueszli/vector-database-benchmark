from caffe2.python.test_util import TestCase
from caffe2.python import workspace, brew
from caffe2.python.model_helper import ModelHelper
from caffe2.python.predictor import mobile_exporter
import numpy as np

class TestMobileExporter(TestCase):

    def test_mobile_exporter(self):
        if False:
            while True:
                i = 10
        model = ModelHelper(name='mobile_exporter_test_model')
        brew.conv(model, 'data', 'conv1', dim_in=1, dim_out=20, kernel=5)
        brew.max_pool(model, 'conv1', 'pool1', kernel=2, stride=2)
        brew.conv(model, 'pool1', 'conv2', dim_in=20, dim_out=50, kernel=5)
        brew.max_pool(model, 'conv2', 'pool2', kernel=2, stride=2)
        brew.fc(model, 'pool2', 'fc3', dim_in=50 * 4 * 4, dim_out=500)
        brew.relu(model, 'fc3', 'fc3')
        brew.fc(model, 'fc3', 'pred', 500, 10)
        brew.softmax(model, 'pred', 'out')
        workspace.RunNetOnce(model.param_init_net)
        (init_net, predict_net) = mobile_exporter.Export(workspace, model.net, model.params)
        np_data = np.random.rand(1, 1, 28, 28).astype(np.float32)
        workspace.FeedBlob('data', np_data)
        workspace.CreateNet(model.net)
        workspace.RunNet(model.net)
        ref_out = workspace.FetchBlob('out')
        workspace.ResetWorkspace()
        workspace.RunNetOnce(init_net)
        workspace.FeedBlob('data', np_data)
        workspace.CreateNet(predict_net, True)
        workspace.RunNet(predict_net.name)
        manual_run_out = workspace.FetchBlob('out')
        np.testing.assert_allclose(ref_out, manual_run_out, atol=1e-10, rtol=1e-10)
        workspace.ResetWorkspace()
        predictor = workspace.Predictor(init_net.SerializeToString(), predict_net.SerializeToString())
        predictor_out = predictor.run([np_data])
        assert len(predictor_out) == 1
        predictor_out = predictor_out[0]
        np.testing.assert_allclose(ref_out, predictor_out, atol=1e-10, rtol=1e-10)

    def test_mobile_exporter_datatypes(self):
        if False:
            for i in range(10):
                print('nop')
        model = ModelHelper(name='mobile_exporter_test_model')
        model.Copy('data_int', 'out')
        model.params.append('data_int')
        model.Copy('data_obj', 'out_obj')
        model.params.append('data_obj')
        workspace.RunNetOnce(model.param_init_net)
        np_data_int = np.random.randint(100, size=(1, 1, 28, 28), dtype=np.int32)
        workspace.FeedBlob('data_int', np_data_int)
        np_data_obj = np.array(['aa', 'bb']).astype(np.dtype('O'))
        workspace.FeedBlob('data_obj', np_data_obj)
        (init_net, predict_net) = mobile_exporter.Export(workspace, model.net, model.params)
        workspace.CreateNet(model.net)
        workspace.RunNet(model.net)
        ref_out = workspace.FetchBlob('out')
        ref_out_obj = workspace.FetchBlob('out_obj')
        workspace.ResetWorkspace()
        workspace.RunNetOnce(init_net)
        workspace.CreateNet(predict_net, True)
        workspace.RunNet(predict_net.name)
        manual_run_out = workspace.FetchBlob('out')
        manual_run_out_obj = workspace.FetchBlob('out_obj')
        np.testing.assert_allclose(ref_out, manual_run_out, atol=1e-10, rtol=1e-10)
        np.testing.assert_equal(ref_out_obj, manual_run_out_obj)
        workspace.ResetWorkspace()
        predictor = workspace.Predictor(init_net.SerializeToString(), predict_net.SerializeToString())
        predictor_out = predictor.run([])
        assert len(predictor_out) == 2
        predictor_out_int = predictor_out[1]
        predictor_out_obj = predictor_out[0]
        if isinstance(predictor_out[1][0], bytes):
            predictor_out_int = predictor_out[0]
            predictor_out_obj = predictor_out[1]
        np.testing.assert_allclose(ref_out, predictor_out_int, atol=1e-10, rtol=1e-10)
        np.testing.assert_equal(ref_out_obj, predictor_out_obj)