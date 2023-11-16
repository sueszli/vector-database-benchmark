import unittest
import unittest.mock as mock
import pickle
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.testing._internal.common_utils import TEST_NUMPY, TemporaryFileName, instantiate_parametrized_tests, run_tests
from torch.testing._internal.common_nn import NNTestCase

class TestPruningNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @unittest.skipIf(not TEST_NUMPY, 'numpy not found')
    def test_validate_pruning_amount_init(self):
        if False:
            print('Hello World!')
        "Test the first util function that validates the pruning\n        amount requested by the user the moment the pruning method\n        is initialized. This test checks that the expected errors are\n        raised whenever the amount is invalid.\n        The original function runs basic type checking + value range checks.\n        It doesn't check the validity of the pruning amount with\n        respect to the size of the tensor to prune. That's left to\n        `_validate_pruning_amount`, tested below.\n        "
        with self.assertRaises(TypeError):
            prune._validate_pruning_amount_init(amount="I'm a string")
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount_init(amount=1.1)
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount_init(amount=20.0)
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount_init(amount=-10)
        prune._validate_pruning_amount_init(amount=0.34)
        prune._validate_pruning_amount_init(amount=1500)
        prune._validate_pruning_amount_init(amount=0)
        prune._validate_pruning_amount_init(amount=0.0)
        prune._validate_pruning_amount_init(amount=1)
        prune._validate_pruning_amount_init(amount=1.0)
        self.assertTrue(True)

    @unittest.skipIf(not TEST_NUMPY, 'numpy not found')
    def test_validate_pruning_amount(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests the second util function that validates the pruning\n        amount requested by the user, this time with respect to the size\n        of the tensor to prune. The rationale is that if the pruning amount,\n        converted to absolute value of units to prune, is larger than\n        the number of units in the tensor, then we expect the util function\n        to raise a value error.\n        '
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount(amount=20, tensor_size=19)
        prune._validate_pruning_amount(amount=0.3, tensor_size=0)
        prune._validate_pruning_amount(amount=19, tensor_size=20)
        prune._validate_pruning_amount(amount=0, tensor_size=0)
        prune._validate_pruning_amount(amount=1, tensor_size=1)
        self.assertTrue(True)

    @unittest.skipIf(not TEST_NUMPY, 'numpy not found')
    def test_compute_nparams_to_prune(self):
        if False:
            return 10
        'Test that requested pruning `amount` gets translated into the\n        correct absolute number of units to prune.\n        '
        self.assertEqual(prune._compute_nparams_toprune(amount=0, tensor_size=15), 0)
        self.assertEqual(prune._compute_nparams_toprune(amount=10, tensor_size=15), 10)
        self.assertEqual(prune._compute_nparams_toprune(amount=1, tensor_size=15), 1)
        self.assertEqual(prune._compute_nparams_toprune(amount=1.0, tensor_size=15), 15)
        self.assertEqual(prune._compute_nparams_toprune(amount=0.4, tensor_size=17), 7)

    def test_random_pruning_sizes(self):
        if False:
            while True:
                i = 10
        'Test that the new parameters and buffers created by the pruning\n        method have the same size as the input tensor to prune. These, in\n        fact, correspond to the pruned version of the tensor itself, its\n        mask, and its original copy, so the size must match.\n        '
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ['weight', 'bias']
        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    original_tensor = getattr(m, name)
                    prune.random_unstructured(m, name=name, amount=0.1)
                    self.assertEqual(original_tensor.size(), getattr(m, name + '_mask').size())
                    self.assertEqual(original_tensor.size(), getattr(m, name + '_orig').size())
                    self.assertEqual(original_tensor.size(), getattr(m, name).size())

    def test_random_pruning_orig(self):
        if False:
            return 10
        "Test that original tensor is correctly stored in 'orig'\n        after pruning is applied. Important to make sure we don't\n        lose info about the original unpruned parameter.\n        "
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ['weight', 'bias']
        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    original_tensor = getattr(m, name)
                    prune.random_unstructured(m, name=name, amount=0.1)
                    self.assertEqual(original_tensor, getattr(m, name + '_orig'))

    def test_random_pruning_new_weight(self):
        if False:
            return 10
        'Test that module.name now contains a pruned version of\n        the original tensor obtained from multiplying it by the mask.\n        '
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ['weight', 'bias']
        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    original_tensor = getattr(m, name)
                    prune.random_unstructured(m, name=name, amount=0.1)
                    self.assertEqual(getattr(m, name), getattr(m, name + '_orig') * getattr(m, name + '_mask').to(dtype=original_tensor.dtype))

    def test_identity_pruning(self):
        if False:
            print('Hello World!')
        'Test that a mask of 1s does not change forward or backward.\n        '
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)
        y_prepruning = m(input_)
        y_prepruning.sum().backward()
        old_grad_weight = m.weight.grad.clone()
        self.assertEqual(old_grad_weight, torch.ones_like(m.weight))
        old_grad_bias = m.bias.grad.clone()
        self.assertEqual(old_grad_bias, torch.ones_like(m.bias))
        m.zero_grad()
        prune.identity(m, name='weight')
        y_postpruning = m(input_)
        self.assertEqual(y_prepruning, y_postpruning)
        y_postpruning.sum().backward()
        self.assertEqual(old_grad_weight, m.weight_orig.grad)
        self.assertEqual(old_grad_bias, m.bias.grad)
        y1 = m(input_)
        y2 = m(input_)
        self.assertEqual(y1, y2)

    def test_random_pruning_0perc(self):
        if False:
            while True:
                i = 10
        'Test that a mask of 1s does not change forward or backward.\n        '
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)
        y_prepruning = m(input_)
        y_prepruning.sum().backward()
        old_grad_weight = m.weight.grad.clone()
        self.assertEqual(old_grad_weight, torch.ones_like(m.weight))
        old_grad_bias = m.bias.grad.clone()
        self.assertEqual(old_grad_bias, torch.ones_like(m.bias))
        m.zero_grad()
        with mock.patch('torch.nn.utils.prune.RandomUnstructured.compute_mask') as compute_mask:
            compute_mask.return_value = torch.ones_like(m.weight)
            prune.random_unstructured(m, name='weight', amount=0.9)
        y_postpruning = m(input_)
        self.assertEqual(y_prepruning, y_postpruning)
        y_postpruning.sum().backward()
        self.assertEqual(old_grad_weight, m.weight_orig.grad)
        self.assertEqual(old_grad_bias, m.bias.grad)
        y1 = m(input_)
        y2 = m(input_)
        self.assertEqual(y1, y2)

    def test_random_pruning(self):
        if False:
            for i in range(10):
                print('nop')
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)
        mask = torch.ones_like(m.weight)
        mask[1, 0] = 0
        mask[0, 3] = 0
        with mock.patch('torch.nn.utils.prune.RandomUnstructured.compute_mask') as compute_mask:
            compute_mask.return_value = mask
            prune.random_unstructured(m, name='weight', amount=0.9)
        y_postpruning = m(input_)
        y_postpruning.sum().backward()
        self.assertEqual(m.weight_orig.grad, mask)
        self.assertEqual(m.bias.grad, torch.ones_like(m.bias))
        old_weight_orig = m.weight_orig.clone()
        learning_rate = 1.0
        for p in m.parameters():
            p.data.sub_(p.grad.data * learning_rate)
        self.assertEqual(old_weight_orig[1, 0], m.weight_orig[1, 0])
        self.assertEqual(old_weight_orig[0, 3], m.weight_orig[0, 3])

    def test_random_pruning_forward(self):
        if False:
            i = 10
            return i + 15
        'check forward with mask (by hand).\n        '
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)
        mask = torch.zeros_like(m.weight)
        mask[1, 0] = 1
        mask[0, 3] = 1
        with mock.patch('torch.nn.utils.prune.RandomUnstructured.compute_mask') as compute_mask:
            compute_mask.return_value = mask
            prune.random_unstructured(m, name='weight', amount=0.9)
        yhat = m(input_)
        self.assertEqual(yhat[0, 0], m.weight_orig[0, 3] + m.bias[0])
        self.assertEqual(yhat[0, 1], m.weight_orig[1, 0] + m.bias[1])

    def test_remove_pruning_forward(self):
        if False:
            return 10
        'Remove pruning and check forward is unchanged from previous\n        pruned state.\n        '
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)
        mask = torch.ones_like(m.weight)
        mask[1, 0] = 0
        mask[0, 3] = 0
        with mock.patch('torch.nn.utils.prune.RandomUnstructured.compute_mask') as compute_mask:
            compute_mask.return_value = mask
            prune.random_unstructured(m, name='weight', amount=0.9)
        y_postpruning = m(input_)
        prune.remove(m, 'weight')
        y_postremoval = m(input_)
        self.assertEqual(y_postpruning, y_postremoval)

    def test_pruning_id_consistency(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that pruning doesn't change the id of the parameters, which\n        would otherwise introduce issues with pre-existing optimizers that\n        point to old parameters.\n        "
        m = nn.Linear(5, 2, bias=False)
        tensor_id = id(list(m.parameters())[0])
        prune.random_unstructured(m, name='weight', amount=0.9)
        self.assertEqual(tensor_id, id(list(m.parameters())[0]))
        prune.remove(m, 'weight')
        self.assertEqual(tensor_id, id(list(m.parameters())[0]))

    def test_random_pruning_pickle(self):
        if False:
            while True:
                i = 10
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ['weight', 'bias']
        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    prune.random_unstructured(m, name=name, amount=0.1)
                    m_new = pickle.loads(pickle.dumps(m))
                    self.assertIsInstance(m_new, type(m))

    def test_multiple_pruning_calls(self):
        if False:
            print('Hello World!')
        m = nn.Conv3d(2, 2, 2)
        prune.l1_unstructured(m, name='weight', amount=0.1)
        weight_mask0 = m.weight_mask
        prune.ln_structured(m, name='weight', amount=0.3, n=2, dim=0)
        hook = next(iter(m._forward_pre_hooks.values()))
        self.assertIsInstance(hook, torch.nn.utils.prune.PruningContainer)
        self.assertEqual(hook._tensor_name, 'weight')
        self.assertEqual(len(hook), 2)
        self.assertIsInstance(hook[0], torch.nn.utils.prune.L1Unstructured)
        self.assertIsInstance(hook[1], torch.nn.utils.prune.LnStructured)
        self.assertTrue(torch.all(m.weight_mask[weight_mask0 == 0] == 0))
        prune.ln_structured(m, name='weight', amount=0.1, n=float('inf'), dim=1)
        hook = next(iter(m._forward_pre_hooks.values()))
        self.assertEqual(hook._tensor_name, 'weight')

    def test_pruning_container(self):
        if False:
            return 10
        container = prune.PruningContainer()
        container._tensor_name = 'test'
        self.assertEqual(len(container), 0)
        p = prune.L1Unstructured(amount=2)
        p._tensor_name = 'test'
        container.add_pruning_method(p)
        q = prune.L1Unstructured(amount=2)
        q._tensor_name = 'another_test'
        with self.assertRaises(ValueError):
            container.add_pruning_method(q)
        with self.assertRaises(TypeError):
            container.add_pruning_method(10)
        with self.assertRaises(TypeError):
            container.add_pruning_method('ugh')

    def test_pruning_container_compute_mask(self):
        if False:
            while True:
                i = 10
        'Test `compute_mask` of pruning container with a known `t` and\n        `default_mask`. Indirectly checks that Ln structured pruning is\n        acting on the right axis.\n        '
        container = prune.PruningContainer()
        container._tensor_name = 'test'
        p = prune.L1Unstructured(amount=2)
        p._tensor_name = 'test'
        container.add_pruning_method(p)
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]], dtype=torch.float32)
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(expected_mask, computed_mask)
        q = prune.LnStructured(amount=1, n=2, dim=0)
        q._tensor_name = 'test'
        container.add_pruning_method(q)
        expected_mask = torch.tensor([[0, 0, 0, 0], [1, 1, 0, 1]], dtype=torch.float32)
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(expected_mask, computed_mask)
        r = prune.LnStructured(amount=1, n=2, dim=1)
        r._tensor_name = 'test'
        container.add_pruning_method(r)
        expected_mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(expected_mask, computed_mask)

    def test_l1_unstructured_pruning(self):
        if False:
            while True:
                i = 10
        'Test that l1 unstructured pruning actually removes the lowest\n        entries by l1 norm (by hand). It also checks that applying l1\n        unstructured pruning more than once respects the previous mask.\n        '
        m = nn.Linear(4, 2)
        m.weight = torch.nn.Parameter(torch.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]], dtype=torch.float32))
        prune.l1_unstructured(m, 'weight', amount=2)
        expected_weight = torch.tensor([[0, 2, 3, 4], [-4, -3, -2, 0]], dtype=m.weight.dtype)
        self.assertEqual(expected_weight, m.weight)
        prune.l1_unstructured(m, 'weight', amount=2)
        expected_weight = torch.tensor([[0, 0, 3, 4], [-4, -3, 0, 0]], dtype=m.weight.dtype)
        self.assertEqual(expected_weight, m.weight)

    def test_l1_unstructured_pruning_with_importance_scores(self):
        if False:
            i = 10
            return i + 15
        'Test that l1 unstructured pruning actually removes the lowest\n        entries of importance scores and not the parameter by l1 norm (by hand).\n        It also checks that applying l1 unstructured pruning more than once\n        respects the previous mask.\n        '
        m = nn.Linear(4, 2)
        m.weight = torch.nn.Parameter(torch.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]], dtype=torch.float32))
        importance_scores = torch.tensor([[4, 2, 1, 3], [-3, -1, -2, -4]], dtype=torch.float32)
        prune.l1_unstructured(m, 'weight', amount=2, importance_scores=importance_scores)
        expected_weight = torch.tensor([[1, 2, 0, 4], [-4, 0, -2, -1]], dtype=m.weight.dtype)
        self.assertEqual(expected_weight, m.weight)
        prune.l1_unstructured(m, 'weight', amount=2, importance_scores=importance_scores)
        expected_weight = torch.tensor([[1, 0, 0, 4], [-4, 0, 0, -1]], dtype=m.weight.dtype)
        self.assertEqual(expected_weight, m.weight)

    def test_unstructured_pruning_same_magnitude(self):
        if False:
            i = 10
            return i + 15
        'Since it may happen that the tensor to prune has entries with the\n        same exact magnitude, it is important to check that pruning happens\n        consistenly based on the bottom % of weights, and not by threshold,\n        which would instead kill off *all* units with magnitude = threshold.\n        '
        AMOUNT = 0.2
        p = prune.L1Unstructured(amount=AMOUNT)
        t = 2 * torch.randint(low=-1, high=2, size=(10, 7))
        nparams_toprune = prune._compute_nparams_toprune(AMOUNT, t.nelement())
        computed_mask = p.compute_mask(t, default_mask=torch.ones_like(t))
        nparams_pruned = torch.sum(computed_mask == 0)
        self.assertEqual(nparams_toprune, nparams_pruned)

    def test_random_structured_pruning_amount(self):
        if False:
            i = 10
            return i + 15
        AMOUNT = 0.6
        AXIS = 2
        p = prune.RandomStructured(amount=AMOUNT, dim=AXIS)
        t = 2 * torch.randint(low=-1, high=2, size=(5, 4, 2)).to(dtype=torch.float32)
        nparams_toprune = prune._compute_nparams_toprune(AMOUNT, t.shape[AXIS])
        computed_mask = p.compute_mask(t, default_mask=torch.ones_like(t))
        remaining_axes = [_ for _ in range(len(t.shape)) if _ != AXIS]
        per_column_sums = sorted(torch.sum(computed_mask == 0, axis=remaining_axes))
        assert per_column_sums == [0, 20]

    def test_ln_structured_pruning(self):
        if False:
            i = 10
            return i + 15
        'Check Ln structured pruning by hand.\n        '
        m = nn.Conv2d(3, 1, 2)
        m.weight.data = torch.tensor([[[[1.0, 2.0], [1.0, 2.5]], [[0.5, 1.0], [0.1, 0.1]], [[-3.0, -5.0], [0.1, -1.0]]]])
        expected_mask_axis1 = torch.ones_like(m.weight)
        expected_mask_axis1[:, 1] = 0.0
        prune.ln_structured(m, 'weight', amount=1, n=2, dim=1)
        self.assertEqual(expected_mask_axis1, m.weight_mask)
        expected_mask_axis3 = expected_mask_axis1
        expected_mask_axis3[:, :, :, 0] = 0.0
        prune.ln_structured(m, 'weight', amount=1, n=1, dim=-1)
        self.assertEqual(expected_mask_axis3, m.weight_mask)

    def test_ln_structured_pruning_importance_scores(self):
        if False:
            print('Hello World!')
        'Check Ln structured pruning by hand.\n        '
        m = nn.Conv2d(3, 1, 2)
        m.weight.data = torch.tensor([[[[1.0, 2.0], [1.0, 2.5]], [[0.5, 1.0], [0.1, 0.1]], [[-3.0, -5.0], [0.1, -1.0]]]])
        importance_scores = torch.tensor([[[[10.0, 1.0], [10.0, 1.0]], [[30.0, 3.0], [30.0, 3.0]], [[-20.0, -2.0], [-20.0, -2.0]]]])
        expected_mask_axis1 = torch.ones_like(m.weight)
        expected_mask_axis1[:, 0] = 0.0
        prune.ln_structured(m, 'weight', amount=1, n=2, dim=1, importance_scores=importance_scores)
        self.assertEqual(expected_mask_axis1, m.weight_mask)
        expected_mask_axis3 = expected_mask_axis1
        expected_mask_axis3[:, :, :, 1] = 0.0
        prune.ln_structured(m, 'weight', amount=1, n=1, dim=-1, importance_scores=importance_scores)
        self.assertEqual(expected_mask_axis3, m.weight_mask)

    def test_remove_pruning(self):
        if False:
            for i in range(10):
                print('nop')
        '`prune.remove` removes the hook and the reparametrization\n        and makes the pruning final in the original parameter.\n        '
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ['weight', 'bias']
        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    prune.random_unstructured(m, name, amount=0.5)
                    self.assertIn(name + '_orig', dict(m.named_parameters()))
                    self.assertIn(name + '_mask', dict(m.named_buffers()))
                    self.assertNotIn(name, dict(m.named_parameters()))
                    self.assertTrue(hasattr(m, name))
                    pruned_t = getattr(m, name)
                    prune.remove(m, name)
                    self.assertIn(name, dict(m.named_parameters()))
                    self.assertNotIn(name + '_orig', dict(m.named_parameters()))
                    self.assertNotIn(name + '_mask', dict(m.named_buffers()))
                    final_t = getattr(m, name)
                    self.assertEqual(pruned_t, final_t)

    def test_remove_pruning_exception(self):
        if False:
            while True:
                i = 10
        'Removing from an unpruned tensor throws an assertion error\n        '
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ['weight', 'bias']
        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    self.assertFalse(prune.is_pruned(m))
                    with self.assertRaises(ValueError):
                        prune.remove(m, name)

    def test_global_pruning(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that global l1 unstructured pruning over 2 parameters removes\n        the `amount=4` smallest global weights across the 2 parameters.\n        '
        m = nn.Linear(4, 2)
        n = nn.Linear(3, 1)
        m.weight = torch.nn.Parameter(torch.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]]).to(dtype=torch.float32))
        n.weight = torch.nn.Parameter(torch.tensor([[0, 0.1, -2]]).to(dtype=torch.float32))
        params_to_prune = ((m, 'weight'), (n, 'weight'))
        prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=4)
        expected_mweight = torch.tensor([[0, 2, 3, 4], [-4, -3, -2, 0]], dtype=m.weight.dtype)
        self.assertEqual(expected_mweight, m.weight)
        expected_nweight = torch.tensor([[0, 0, -2]]).to(dtype=n.weight.dtype)
        self.assertEqual(expected_nweight, n.weight)

    def test_global_pruning_importance_scores(self):
        if False:
            i = 10
            return i + 15
        'Test that global l1 unstructured pruning over 2 parameters removes\n        the `amount=4` smallest global weights across the 2 parameters.\n        '
        m = nn.Linear(4, 2)
        n = nn.Linear(3, 1)
        m.weight = torch.nn.Parameter(torch.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]]).to(dtype=torch.float32))
        m_importance_scores = torch.tensor([[4, 2, 1, 3], [-3, -1, -2, -4]], dtype=torch.float32)
        n.weight = torch.nn.Parameter(torch.tensor([[0, 0.1, -2]]).to(dtype=torch.float32))
        n_importance_scores = torch.tensor([[0, 10.0, -0.2]]).to(dtype=torch.float32)
        params_to_prune = ((m, 'weight'), (n, 'weight'))
        importance_scores = {(m, 'weight'): m_importance_scores, (n, 'weight'): n_importance_scores}
        prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=4, importance_scores=importance_scores)
        expected_m_weight = torch.tensor([[1, 2, 0, 4], [-4, 0, -2, -1]], dtype=m.weight.dtype)
        self.assertEqual(expected_m_weight, m.weight)
        expected_n_weight = torch.tensor([[0, 0.1, 0]]).to(dtype=n.weight.dtype)
        self.assertEqual(expected_n_weight, n.weight)

    def test_custom_from_mask_pruning(self):
        if False:
            return 10
        'Test that the CustomFromMask is capable of receiving\n        as input at instantiation time a custom mask, and combining it with\n        the previous default mask to generate the correct final mask.\n        '
        mask = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]])
        default_mask = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
        t = torch.rand_like(mask.to(dtype=torch.float32))
        p = prune.CustomFromMask(mask=mask)
        computed_mask = p.compute_mask(t, default_mask)
        expected_mask = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1]], dtype=computed_mask.dtype)
        self.assertEqual(computed_mask, expected_mask)

    def test_pruning_rollback(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that if something fails when the we try to compute the mask,\n        then the model isn't left in some intermediate half-pruned state.\n        The try/except statement in `apply` should handle rolling back\n        to the previous state before pruning began.\n        "
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ['weight', 'bias']
        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    with mock.patch('torch.nn.utils.prune.L1Unstructured.compute_mask') as compute_mask:
                        compute_mask.side_effect = Exception('HA!')
                        with self.assertRaises(Exception):
                            prune.l1_unstructured(m, name=name, amount=0.9)
                        self.assertTrue(name in dict(m.named_parameters()))
                        self.assertFalse(name + '_mask' in dict(m.named_buffers()))
                        self.assertFalse(name + '_orig' in dict(m.named_parameters()))

    def test_pruning_serialization_model(self):
        if False:
            for i in range(10):
                print('nop')
        model = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1))
        self.assertNotIn('0.weight_orig', model.state_dict())
        self.assertNotIn('0.weight_mask', model.state_dict())
        self.assertIn('0.weight', model.state_dict())
        prune.l1_unstructured(module=model[0], name='weight', amount=0.9)
        self.assertIn('0.weight_orig', model.state_dict())
        self.assertIn('0.weight_mask', model.state_dict())
        self.assertNotIn('0.weight', model.state_dict())
        self.assertTrue(hasattr(model[0], 'weight'))
        pruned_weight = model[0].weight
        with TemporaryFileName() as fname:
            torch.save(model, fname)
            new_model = torch.load(fname)
        self.assertIn('0.weight_orig', new_model.state_dict())
        self.assertIn('0.weight_mask', new_model.state_dict())
        self.assertNotIn('0.weight', new_model.state_dict())
        self.assertTrue(hasattr(new_model[0], 'weight'))
        self.assertEqual(pruned_weight, new_model[0].weight)

    def test_pruning_serialization_state_dict(self):
        if False:
            print('Hello World!')
        model = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1))
        self.assertNotIn('0.weight_orig', model.state_dict())
        self.assertNotIn('0.weight_mask', model.state_dict())
        self.assertIn('0.weight', model.state_dict())
        prune.l1_unstructured(module=model[0], name='weight', amount=0.9)
        self.assertIn('0.weight_orig', model.state_dict())
        self.assertIn('0.weight_mask', model.state_dict())
        self.assertNotIn('0.weight', model.state_dict())
        self.assertTrue(hasattr(model[0], 'weight'))
        pruned_weight = model[0].weight
        prune.remove(module=model[0], name='weight')
        self.assertNotIn('0.weight_orig', model.state_dict())
        self.assertNotIn('0.weight_mask', model.state_dict())
        self.assertIn('0.weight', model.state_dict())
        new_model = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1))
        with TemporaryFileName() as fname:
            torch.save(model.state_dict(), fname)
            new_model.load_state_dict(torch.load(fname))
        self.assertNotIn('0.weight_orig', new_model.state_dict())
        self.assertNotIn('0.weight_mask', new_model.state_dict())
        self.assertIn('0.weight', new_model.state_dict())
        self.assertEqual(pruned_weight, new_model[0].weight)

    def test_prune(self):
        if False:
            i = 10
            return i + 15
        p = prune.L1Unstructured(amount=2)
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]])
        pruned_tensor = p.prune(t, default_mask)
        self.assertEqual(t * expected_mask, pruned_tensor)

    def test_prune_importance_scores(self):
        if False:
            for i in range(10):
                print('nop')
        p = prune.L1Unstructured(amount=2)
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        importance_scores = torch.tensor([[1, 2, 3, 4], [1.5, 1.6, 1.7, 1.8]]).to(dtype=torch.float32)
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        expected_mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 1]])
        pruned_tensor = p.prune(t, default_mask, importance_scores=importance_scores)
        self.assertEqual(t * expected_mask, pruned_tensor)

    def test_prune_importance_scores_mimic_default(self):
        if False:
            i = 10
            return i + 15
        p = prune.L1Unstructured(amount=2)
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]])
        pruned_tensor_without_importance_scores = p.prune(t, default_mask)
        pruned_tensor_with_importance_scores = p.prune(t, default_mask, importance_scores=t)
        self.assertEqual(pruned_tensor_without_importance_scores, pruned_tensor_with_importance_scores)
        self.assertEqual(t * expected_mask, pruned_tensor_without_importance_scores)

    def test_rnn_pruning(self):
        if False:
            while True:
                i = 10
        l = torch.nn.LSTM(32, 32)
        prune.l1_unstructured(l, 'weight_ih_l0', 0.5)
        assert sum([isinstance(p, torch.nn.Parameter) for p in l._flat_weights]) == 3
        prune.remove(l, 'weight_ih_l0')
        assert sum([isinstance(p, torch.nn.Parameter) for p in l._flat_weights]) == 4
        assert 'weight_ih_l0' in l._parameters
        assert l._parameters['weight_ih_l0'] is not None
        assert 'weight_ih_l0_orig' not in l._parameters
        assert 'weight_ih_l0' in dict(l.named_parameters())
        assert dict(l.named_parameters())['weight_ih_l0'] is not None
        assert 'weight_ih_l0_orig' not in dict(l.named_parameters())
instantiate_parametrized_tests(TestPruningNN)
if __name__ == '__main__':
    run_tests()