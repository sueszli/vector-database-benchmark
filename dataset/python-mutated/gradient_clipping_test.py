import unittest
from caffe2.python import workspace, brew, model_helper
from caffe2.python.modeling.gradient_clipping import GradientClipping
import numpy as np

class GradientClippingTest(unittest.TestCase):

    def test_gradient_clipping_by_norm(self):
        if False:
            while True:
                i = 10
        model = model_helper.ModelHelper(name='test')
        data = model.net.AddExternalInput('data')
        fc1 = brew.fc(model, data, 'fc1', dim_in=4, dim_out=2)
        fc2 = brew.fc(model, fc1, 'fc2', dim_in=2, dim_out=1)
        sigm = model.net.Sigmoid(fc2, 'sigm')
        sq = model.net.SquaredL2Distance([sigm, 'label'], 'sq')
        loss = model.net.SumElements(sq, 'loss')
        grad_map = model.AddGradientOperators([loss])
        grad_map_for_param = {key: grad_map[key] for key in ['fc1_w', 'fc2_w']}
        net_modifier = GradientClipping(grad_clip_method='by_norm', clip_norm_type='l2_norm', clip_threshold=0.1)
        net_modifier(model.net, grad_map=grad_map_for_param)
        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))
        workspace.FeedBlob('label', np.random.rand(10, 1).astype(np.float32))
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        self.assertEqual(len(model.net.Proto().op), 17)

    def test_gradient_clipping_by_norm_l1_norm(self):
        if False:
            while True:
                i = 10
        model = model_helper.ModelHelper(name='test')
        data = model.net.AddExternalInput('data')
        fc1 = brew.fc(model, data, 'fc1', dim_in=4, dim_out=2)
        fc2 = brew.fc(model, fc1, 'fc2', dim_in=2, dim_out=1)
        sigm = model.net.Sigmoid(fc2, 'sigm')
        sq = model.net.SquaredL2Distance([sigm, 'label'], 'sq')
        loss = model.net.SumElements(sq, 'loss')
        grad_map = model.AddGradientOperators([loss])
        grad_map_for_param = {key: grad_map[key] for key in ['fc1_w', 'fc2_w']}
        net_modifier = GradientClipping(grad_clip_method='by_norm', clip_norm_type='l1_norm', clip_threshold=0.1)
        net_modifier(model.net, grad_map=grad_map_for_param)
        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))
        workspace.FeedBlob('label', np.random.rand(10, 1).astype(np.float32))
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        self.assertEqual(len(model.net.Proto().op), 15)

    def test_gradient_clipping_by_norm_using_param_norm(self):
        if False:
            print('Hello World!')
        model = model_helper.ModelHelper(name='test')
        data = model.net.AddExternalInput('data')
        fc1 = brew.fc(model, data, 'fc1', dim_in=4, dim_out=2)
        fc2 = brew.fc(model, fc1, 'fc2', dim_in=2, dim_out=1)
        sigm = model.net.Sigmoid(fc2, 'sigm')
        sq = model.net.SquaredL2Distance([sigm, 'label'], 'sq')
        loss = model.net.SumElements(sq, 'loss')
        grad_map = model.AddGradientOperators([loss])
        grad_map_for_param = {key: grad_map[key] for key in ['fc1_w', 'fc2_w']}
        net_modifier = GradientClipping(grad_clip_method='by_norm', clip_norm_type='l2_norm', clip_threshold=0.1, use_parameter_norm=True)
        net_modifier(model.net, grad_map=grad_map_for_param)
        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))
        workspace.FeedBlob('label', np.random.rand(10, 1).astype(np.float32))
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        self.assertEqual(len(model.net.Proto().op), 21)

    def test_gradient_clipping_by_norm_compute_norm_ratio(self):
        if False:
            while True:
                i = 10
        model = model_helper.ModelHelper(name='test')
        data = model.net.AddExternalInput('data')
        fc1 = brew.fc(model, data, 'fc1', dim_in=4, dim_out=2)
        fc2 = brew.fc(model, fc1, 'fc2', dim_in=2, dim_out=1)
        sigm = model.net.Sigmoid(fc2, 'sigm')
        sq = model.net.SquaredL2Distance([sigm, 'label'], 'sq')
        loss = model.net.SumElements(sq, 'loss')
        grad_map = model.AddGradientOperators([loss])
        grad_map_for_param = {key: grad_map[key] for key in ['fc1_w', 'fc2_w']}
        net_modifier = GradientClipping(grad_clip_method='by_norm', clip_norm_type='l2_norm', clip_threshold=0.1, use_parameter_norm=True, compute_norm_ratio=True)
        net_modifier(model.net, grad_map=grad_map_for_param)
        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))
        workspace.FeedBlob('label', np.random.rand(10, 1).astype(np.float32))
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        self.assertEqual(len(model.net.Proto().op), 23)

    def test_gradient_clipping_by_value(self):
        if False:
            while True:
                i = 10
        model = model_helper.ModelHelper(name='test')
        data = model.net.AddExternalInput('data')
        fc1 = brew.fc(model, data, 'fc1', dim_in=4, dim_out=2)
        fc2 = brew.fc(model, fc1, 'fc2', dim_in=2, dim_out=1)
        sigm = model.net.Sigmoid(fc2, 'sigm')
        sq = model.net.SquaredL2Distance([sigm, 'label'], 'sq')
        loss = model.net.SumElements(sq, 'loss')
        grad_map = model.AddGradientOperators([loss])
        grad_map_for_param = {key: grad_map[key] for key in ['fc1_w', 'fc2_w']}
        clip_max = 1e-08
        clip_min = 0
        net_modifier = GradientClipping(grad_clip_method='by_value', clip_max=clip_max, clip_min=clip_min)
        net_modifier(model.net, grad_map=grad_map_for_param)
        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))
        workspace.FeedBlob('label', np.random.rand(10, 1).astype(np.float32))
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        self.assertEqual(len(model.net.Proto().op), 13)
        fc1_w_grad = workspace.FetchBlob('fc1_w_grad')
        self.assertLessEqual(np.amax(fc1_w_grad), clip_max)
        self.assertGreaterEqual(np.amin(fc1_w_grad), clip_min)

    def test_gradient_clipping_by_norm_including_blobs(self):
        if False:
            while True:
                i = 10
        model = model_helper.ModelHelper(name='test')
        data = model.net.AddExternalInput('data')
        fc1 = brew.fc(model, data, 'fc1', dim_in=4, dim_out=2)
        fc2 = brew.fc(model, fc1, 'fc2', dim_in=2, dim_out=1)
        sigm = model.net.Sigmoid(fc2, 'sigm')
        sq = model.net.SquaredL2Distance([sigm, 'label'], 'sq')
        loss = model.net.SumElements(sq, 'loss')
        grad_map = model.AddGradientOperators([loss])
        grad_map_for_param = {key: grad_map[key] for key in ['fc1_w', 'fc2_w']}
        net_modifier = GradientClipping(grad_clip_method='by_norm', clip_norm_type='l2_norm', clip_threshold=0.1, blobs_to_include=['fc1_w'], blobs_to_exclude=None)
        net_modifier(model.net, grad_map=grad_map_for_param)
        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))
        workspace.FeedBlob('label', np.random.rand(10, 1).astype(np.float32))
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        self.assertEqual(len(model.net.Proto().op), 14)

    def test_gradient_clipping_by_norm_excluding_blobs(self):
        if False:
            return 10
        model = model_helper.ModelHelper(name='test')
        data = model.net.AddExternalInput('data')
        fc1 = brew.fc(model, data, 'fc1', dim_in=4, dim_out=2)
        fc2 = brew.fc(model, fc1, 'fc2', dim_in=2, dim_out=1)
        sigm = model.net.Sigmoid(fc2, 'sigm')
        sq = model.net.SquaredL2Distance([sigm, 'label'], 'sq')
        loss = model.net.SumElements(sq, 'loss')
        grad_map = model.AddGradientOperators([loss])
        grad_map_for_param = {key: grad_map[key] for key in ['fc1_w', 'fc2_w']}
        net_modifier = GradientClipping(grad_clip_method='by_norm', clip_norm_type='l2_norm', clip_threshold=0.1, blobs_to_include=None, blobs_to_exclude=['fc1_w', 'fc2_w'])
        net_modifier(model.net, grad_map=grad_map_for_param)
        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))
        workspace.FeedBlob('label', np.random.rand(10, 1).astype(np.float32))
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        self.assertEqual(len(model.net.Proto().op), 11)