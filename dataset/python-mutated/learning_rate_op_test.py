from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import given, settings
import hypothesis.strategies as st
import copy
from functools import partial
import math
import numpy as np

class TestLearningRate(serial.SerializedTestCase):

    @given(**hu.gcs_cpu_only)
    @settings(deadline=None, max_examples=50)
    def test_alter_learning_rate_op(self, gc, dc):
        if False:
            for i in range(10):
                print('nop')
        iter = np.random.randint(low=1, high=100000.0, size=1)
        active_period = int(np.random.randint(low=1, high=1000.0, size=1))
        inactive_period = int(np.random.randint(low=1, high=1000.0, size=1))
        base_lr = float(np.random.random(1))

        def ref(iter):
            if False:
                for i in range(10):
                    print('nop')
            iter = float(iter)
            reminder = iter % (active_period + inactive_period)
            if reminder < active_period:
                return (np.array(base_lr),)
            else:
                return (np.array(0.0),)
        op = core.CreateOperator('LearningRate', 'iter', 'lr', policy='alter', active_first=True, base_lr=base_lr, active_period=active_period, inactive_period=inactive_period)
        self.assertReferenceChecks(gc, op, [iter], ref)

    @given(**hu.gcs_cpu_only)
    def test_hill_learning_rate_op(self, gc, dc):
        if False:
            print('Hello World!')
        iter = np.random.randint(low=1, high=100000.0, size=1)
        num_iter = int(np.random.randint(low=100.0, high=100000000.0, size=1))
        start_multiplier = 0.0001
        gamma = 1.0
        power = 0.5
        end_multiplier = 0.01
        base_lr = float(np.random.random(1))

        def ref(iter):
            if False:
                for i in range(10):
                    print('nop')
            iter = float(iter)
            if iter < num_iter:
                lr = start_multiplier + (1.0 - start_multiplier) * iter / num_iter
            else:
                iter -= num_iter
                lr = math.pow(1.0 + gamma * iter, -power)
                lr = max(lr, end_multiplier)
            return (np.array(base_lr * lr),)
        op = core.CreateOperator('LearningRate', 'data', 'out', policy='hill', base_lr=base_lr, num_iter=num_iter, start_multiplier=start_multiplier, gamma=gamma, power=power, end_multiplier=end_multiplier)
        self.assertReferenceChecks(gc, op, [iter], ref)

    @given(**hu.gcs_cpu_only)
    def test_slope_learning_rate_op(self, gc, dc):
        if False:
            print('Hello World!')
        iter = np.random.randint(low=1, high=100000.0, size=1)
        num_iter_1 = int(np.random.randint(low=100.0, high=1000.0, size=1))
        multiplier_1 = 1.0
        num_iter_2 = num_iter_1 + int(np.random.randint(low=100.0, high=1000.0, size=1))
        multiplier_2 = 0.5
        base_lr = float(np.random.random(1))

        def ref(iter):
            if False:
                return 10
            iter = float(iter)
            if iter < num_iter_1:
                lr = multiplier_1
            else:
                lr = max(multiplier_1 + (iter - num_iter_1) * (multiplier_2 - multiplier_1) / (num_iter_2 - num_iter_1), multiplier_2)
            return (np.array(base_lr * lr),)
        op = core.CreateOperator('LearningRate', 'data', 'out', policy='slope', base_lr=base_lr, num_iter_1=num_iter_1, multiplier_1=multiplier_1, num_iter_2=num_iter_2, multiplier_2=multiplier_2)
        self.assertReferenceChecks(gc, op, [iter], ref)

    @given(**hu.gcs_cpu_only)
    @settings(max_examples=10)
    def test_gate_learningrate(self, gc, dc):
        if False:
            for i in range(10):
                print('nop')
        iter = np.random.randint(low=1, high=100000.0, size=1)
        num_iter = int(np.random.randint(low=100.0, high=1000.0, size=1))
        base_lr = float(np.random.uniform(-1, 1))
        multiplier_1 = float(np.random.uniform(-1, 1))
        multiplier_2 = float(np.random.uniform(-1, 1))

        def ref(iter):
            if False:
                for i in range(10):
                    print('nop')
            iter = float(iter)
            if iter < num_iter:
                return (np.array(multiplier_1 * base_lr),)
            else:
                return (np.array(multiplier_2 * base_lr),)
        op = core.CreateOperator('LearningRate', 'data', 'out', policy='gate', num_iter=num_iter, multiplier_1=multiplier_1, multiplier_2=multiplier_2, base_lr=base_lr)
        self.assertReferenceChecks(gc, op, [iter], ref)

    @given(gc=hu.gcs['gc'], min_num_iter=st.integers(min_value=10, max_value=20), max_num_iter=st.integers(min_value=50, max_value=100))
    @settings(max_examples=2, deadline=None)
    def test_composite_learning_rate_op(self, gc, min_num_iter, max_num_iter):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(65535)
        num_lr_policy = 4
        iter_nums = np.random.randint(low=min_num_iter, high=max_num_iter, size=num_lr_policy)
        accu_iter_num = copy.deepcopy(iter_nums)
        for i in range(1, num_lr_policy):
            accu_iter_num[i] += accu_iter_num[i - 1]
        total_iter_nums = accu_iter_num[-1]
        policy_lr_scale = np.random.uniform(low=0.1, high=2.0, size=num_lr_policy)
        step_size = np.random.randint(low=2, high=min_num_iter // 2)
        step_gamma = np.random.random()
        exp_gamma = np.random.random()
        base_lr = 0.1

        def step_lr(iter, lr_scale):
            if False:
                return 10
            return math.pow(step_gamma, iter // step_size) * lr_scale

        def exp_lr(iter, lr_scale):
            if False:
                print('Hello World!')
            return math.pow(exp_gamma, iter) * lr_scale

        def fixed_lr(iter, lr_scale):
            if False:
                print('Hello World!')
            return lr_scale

        def one_policy_check_ref(iter, lr_scale):
            if False:
                print('Hello World!')
            iter = int(iter)
            exp_lr_val = exp_lr(iter, lr_scale=lr_scale)
            return (np.array(base_lr * exp_lr_val),)
        op = core.CreateOperator('LearningRate', 'data', 'out', policy='composite', sub_policy_num_iters=iter_nums[:1], sub_policy_0_lr_scale=policy_lr_scale[0], sub_policy_0_policy='exp', sub_policy_0_gamma=exp_gamma, base_lr=base_lr)
        for iter_idx in range(1, total_iter_nums + 1):
            self.assertReferenceChecks(gc, op, [np.asarray([iter_idx])], partial(one_policy_check_ref, lr_scale=policy_lr_scale[0]))

        def all_sub_policy_check_ref(iter, lr_scale):
            if False:
                print('Hello World!')
            assert iter <= accu_iter_num[3]
            if iter <= accu_iter_num[0]:
                lr = exp_lr(iter, lr_scale=lr_scale)
            elif iter <= accu_iter_num[1]:
                lr = step_lr(iter, lr_scale=lr_scale)
            elif iter <= accu_iter_num[2]:
                lr = fixed_lr(iter, lr_scale=lr_scale)
            else:
                lr = exp_lr(iter, lr_scale=lr_scale)
            return (np.array(base_lr * lr),)
        op = core.CreateOperator('LearningRate', 'data', 'out', policy='composite', sub_policy_num_iters=iter_nums, sub_policy_0_policy='exp', sub_policy_0_lr_scale=policy_lr_scale[0], sub_policy_0_gamma=exp_gamma, sub_policy_1_policy='step', sub_policy_1_lr_scale=policy_lr_scale[1], sub_policy_1_stepsize=step_size, sub_policy_1_gamma=step_gamma, sub_policy_2_policy='fixed', sub_policy_2_lr_scale=policy_lr_scale[2], sub_policy_3_policy='exp', sub_policy_3_gamma=exp_gamma, sub_policy_3_lr_scale=policy_lr_scale[3], base_lr=base_lr)
        iter_policy = 0
        for iter_idx in range(1, total_iter_nums + 1):
            if iter_idx > accu_iter_num[iter_policy]:
                iter_policy += 1
            self.assertReferenceChecks(gc, op, [np.asarray([iter_idx])], partial(all_sub_policy_check_ref, lr_scale=policy_lr_scale[iter_policy]))
if __name__ == '__main__':
    import unittest
    unittest.main()