import random
import unittest
import numpy as np
import torch
from snorkel.slicing import SliceCombinerModule

class SliceCombinerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

    def test_forward_shape(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the reweight representation shape matches expected feature size.'
        batch_size = 4
        h_dim = 20
        num_classes = 2
        outputs = {'task_slice:base_ind_head': torch.FloatTensor(batch_size, 2).uniform_(0, 1), 'task_slice:base_pred_transform': torch.FloatTensor(batch_size, h_dim).uniform_(0, 1), 'task_slice:base_pred_head': torch.FloatTensor(batch_size, num_classes).uniform_(0, 1)}
        combiner_module = SliceCombinerModule()
        combined_rep = combiner_module(outputs)
        self.assertEqual(tuple(combined_rep.shape), (batch_size, h_dim))
        batch_size = 1
        h_dim = 1
        num_classes = 2
        outputs = {'task_slice:base_ind_head': torch.FloatTensor(batch_size, 2).uniform_(0, 1), 'task_slice:base_pred_transform': torch.FloatTensor(batch_size, h_dim).uniform_(0, 1), 'task_slice:base_pred_head': torch.FloatTensor(batch_size, num_classes).uniform_(0, 1)}
        combiner_module = SliceCombinerModule()
        combined_rep = combiner_module(outputs)
        self.assertEqual(tuple(combined_rep.shape), (batch_size, h_dim))
        batch_size = 1
        h_dim = 1
        num_classes = 1
        outputs = {'task_slice:base_ind_head': torch.FloatTensor(batch_size, 2).uniform_(0, 1), 'task_slice:base_pred_transform': torch.FloatTensor(batch_size, h_dim).uniform_(0, 1), 'task_slice:base_pred_head': torch.FloatTensor(batch_size, num_classes).uniform_(0, 1)}
        combiner_module = SliceCombinerModule()
        with self.assertRaisesRegex(NotImplementedError, 'requires output shape'):
            combined_rep = combiner_module(outputs)

    def test_average_reweighting(self):
        if False:
            print('Hello World!')
        'Test average reweighting (equal weight across two slices).'
        batch_size = 4
        h_dim = 20
        num_classes = 2
        outputs = {'task_slice:a_ind_head': torch.ones(batch_size, 2) * 10.0, 'task_slice:a_pred_transform': torch.ones(batch_size, h_dim) * 4, 'task_slice:a_pred_head': torch.ones(batch_size, num_classes) * 10.0, 'task_slice:b_ind_head': torch.ones(batch_size, 2) * -10.0, 'task_slice:b_pred_transform': torch.ones(batch_size, h_dim) * 2, 'task_slice:b_pred_head': torch.ones(batch_size, num_classes) * -10.0}
        combiner_module = SliceCombinerModule()
        combined_rep = combiner_module(outputs)
        self.assertTrue(torch.allclose(combined_rep, torch.ones(batch_size, h_dim) * 3))

    def test_average_reweighting_by_ind(self):
        if False:
            print('Hello World!')
        'Test average reweighting by ind (zeros in pred head).'
        batch_size = 4
        h_dim = 20
        num_classes = 2
        outputs = {'task_slice:a_ind_head': torch.ones(batch_size, 2) * 10.0, 'task_slice:a_pred_transform': torch.ones(batch_size, h_dim) * 4, 'task_slice:a_pred_head': torch.zeros(batch_size, num_classes), 'task_slice:b_ind_head': torch.ones(batch_size, 2) * -10.0, 'task_slice:b_pred_transform': torch.ones(batch_size, h_dim) * 2, 'task_slice:b_pred_head': torch.zeros(batch_size, num_classes)}
        combiner_module = SliceCombinerModule()
        combined_rep = combiner_module(outputs)
        self.assertTrue(torch.allclose(combined_rep, torch.ones(batch_size, h_dim) * 3))

    def test_average_reweighting_by_pred_confidence(self):
        if False:
            print('Hello World!')
        'Test average reweighting by pred confidence (zeros in ind head).'
        batch_size = 4
        h_dim = 20
        num_classes = 2
        outputs = {'task_slice:a_ind_head': torch.zeros(batch_size, 2), 'task_slice:a_pred_transform': torch.ones(batch_size, h_dim) * 4, 'task_slice:a_pred_head': torch.ones(batch_size, num_classes) * 5, 'task_slice:b_ind_head': torch.zeros(batch_size, 2), 'task_slice:b_pred_transform': torch.ones(batch_size, h_dim) * 2, 'task_slice:b_pred_head': torch.ones(batch_size, num_classes) * 5}
        combiner_module = SliceCombinerModule()
        combined_rep = combiner_module(outputs)
        self.assertTrue(torch.all(combined_rep == torch.ones(batch_size, h_dim) * 3))
        outputs = {'task_slice:a_ind_head': torch.zeros(batch_size, 2), 'task_slice:a_pred_transform': torch.ones(batch_size, h_dim) * 4, 'task_slice:a_pred_head': torch.ones(batch_size, num_classes) * -5, 'task_slice:b_ind_head': torch.zeros(batch_size, 2), 'task_slice:b_pred_transform': torch.ones(batch_size, h_dim) * 2, 'task_slice:b_pred_head': torch.ones(batch_size, num_classes) * 5}
        combiner_module = SliceCombinerModule()
        combined_rep = combiner_module(outputs)
        self.assertTrue(torch.allclose(combined_rep, torch.ones(batch_size, h_dim) * 3))

    def test_many_slices(self):
        if False:
            print('Hello World!')
        'Test combiner on 100 synthetic generated slices.'
        batch_size = 4
        h_dim = 20
        num_classes = 2
        outputs = {}
        for i in range(100):
            if i % 2 == 0:
                outputs[f'task_slice:{i}_ind_head'] = torch.ones(batch_size, 2) * 20.0
                outputs[f'task_slice:{i}_pred_transform'] = torch.ones(batch_size, h_dim) * 4
                outputs[f'task_slice:{i}_pred_head'] = torch.ones(batch_size, num_classes) * 20.0
            else:
                outputs[f'task_slice:{i}_ind_head'] = torch.ones(batch_size, 2) * -20.0
                outputs[f'task_slice:{i}_pred_transform'] = torch.ones(batch_size, h_dim) * 2
                outputs[f'task_slice:{i}_pred_head'] = torch.ones(batch_size, num_classes) * -20.0
        combiner_module = SliceCombinerModule()
        combined_rep = combiner_module(outputs)
        self.assertTrue(torch.allclose(combined_rep, torch.ones(batch_size, h_dim) * 3))

    def test_combiner_multiclass(self):
        if False:
            for i in range(10):
                print('nop')
        'Test combiner in multiclass setting.'
        batch_size = 4
        h_dim = 20
        num_classes = 10
        max_score_indexes_a = [random.randint(0, num_classes) for _ in range(batch_size)]
        pred_outputs_a = torch.FloatTensor(batch_size, num_classes).uniform_(-5, 5)
        pred_outputs_a = pred_outputs_a.scatter_(1, torch.tensor(max_score_indexes_a).unsqueeze(1), 10.0)
        max_score_indexes_b = [random.randint(0, num_classes) for _ in range(batch_size)]
        pred_outputs_b = torch.FloatTensor(batch_size, num_classes).uniform_(-5, 5)
        pred_outputs_b = pred_outputs_b.scatter_(1, torch.tensor(max_score_indexes_b).unsqueeze(1), 10.0)
        outputs = {'task_slice:a_ind_head': torch.ones(batch_size, 2) * -10.0, 'task_slice:a_pred_transform': torch.ones(batch_size, h_dim) * 4, 'task_slice:a_pred_head': pred_outputs_a, 'task_slice:b_ind_head': torch.ones(batch_size, 2) * 10.0, 'task_slice:b_pred_transform': torch.ones(batch_size, h_dim) * 2, 'task_slice:b_pred_head': pred_outputs_b}
        combiner_module = SliceCombinerModule()
        with self.assertRaisesRegex(NotImplementedError, 'more than 2 classes'):
            combiner_module(outputs)

    def test_temperature(self):
        if False:
            for i in range(10):
                print('nop')
        'Test temperature parameter for attention weights.'
        batch_size = 4
        h_dim = 20
        num_classes = 2
        epsilon = 1e-05
        outputs = {'task_slice:a_ind_head': torch.ones(batch_size, 2) * 10.0 + torch.FloatTensor(batch_size, 2).normal_(0.0, epsilon), 'task_slice:a_pred_transform': torch.ones(batch_size, h_dim) * 4 + torch.FloatTensor(batch_size, h_dim).normal_(0.0, epsilon), 'task_slice:a_pred_head': torch.ones(batch_size, num_classes) * 10.0 + torch.FloatTensor(batch_size, num_classes).normal_(0.0, epsilon), 'task_slice:b_ind_head': torch.ones(batch_size, 2) * -10.0 + torch.FloatTensor(batch_size, 2).normal_(0.0, epsilon), 'task_slice:b_pred_transform': torch.ones(batch_size, h_dim) * 2 + torch.FloatTensor(batch_size, h_dim).normal_(0.0, epsilon), 'task_slice:b_pred_head': torch.ones(batch_size, num_classes) * -10.0 + torch.FloatTensor(batch_size, num_classes).normal_(0.0, epsilon)}
        combiner_module = SliceCombinerModule(temperature=100000.0)
        combined_rep = combiner_module(outputs)
        self.assertTrue(torch.allclose(combined_rep, torch.ones(batch_size, h_dim) * 3))
        combiner_module = SliceCombinerModule(temperature=1e-15)
        combined_rep = combiner_module(outputs)
        isclose_four = torch.isclose(combined_rep, torch.ones(batch_size, h_dim) * 2, atol=0.0001)
        isclose_two = torch.isclose(combined_rep, torch.ones(batch_size, h_dim) * 4, atol=0.0001)
        num_matching_original = torch.sum(isclose_four) + torch.sum(isclose_two)
        self.assertEqual(num_matching_original, batch_size * h_dim)
if __name__ == '__main__':
    unittest.main()