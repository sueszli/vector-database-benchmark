import unittest
from transformers import is_torch_available
from transformers.testing_utils import require_torch
if is_torch_available():
    import torch
    from transformers.generation import DisjunctiveConstraint

@require_torch
class ConstraintTest(unittest.TestCase):

    def test_input_types(self):
        if False:
            return 10
        cset = [[1, 2, 4], [1, 2, 3, 4]]
        dc = DisjunctiveConstraint(cset)
        self.assertTrue(isinstance(dc.token_ids, list))
        with self.assertRaises(ValueError):
            DisjunctiveConstraint(torch.LongTensor([[1, 2, 4], [1, 2, 3]]))
        with self.assertRaises(ValueError):
            DisjunctiveConstraint([torch.LongTensor([1, 2, 4]), torch.LongTensor([1, 2, 3, 4, 5])])

    def test_check_illegal_input(self):
        if False:
            return 10
        cset = [[1, 2], [1, 2, 3, 4]]
        with self.assertRaises(ValueError):
            DisjunctiveConstraint(cset)

    def test_example_progression(self):
        if False:
            print('Hello World!')
        cset = [[1, 2, 3], [1, 2, 4]]
        dc = DisjunctiveConstraint(cset)
        (stepped, completed, reset) = dc.update(1)
        desired = stepped is True and completed is False and (reset is False)
        self.assertTrue(desired)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1])
        (stepped, completed, reset) = dc.update(2)
        desired = stepped is True and completed is False and (reset is False)
        self.assertTrue(desired)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1, 2])
        (stepped, completed, reset) = dc.update(3)
        desired = stepped is True and completed is True and (reset is False)
        self.assertTrue(desired)
        self.assertTrue(dc.completed)
        self.assertTrue(dc.current_seq == [1, 2, 3])

    def test_example_progression_unequal_three_mid_and_reset(self):
        if False:
            for i in range(10):
                print('nop')
        cset = [[1, 2, 3], [1, 2, 4, 5], [1, 2, 5]]
        dc = DisjunctiveConstraint(cset)
        (stepped, completed, reset) = dc.update(1)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1])
        (stepped, completed, reset) = dc.update(2)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1, 2])
        (stepped, completed, reset) = dc.update(4)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1, 2, 4])
        (stepped, completed, reset) = dc.update(5)
        self.assertTrue(dc.completed)
        self.assertTrue(dc.current_seq == [1, 2, 4, 5])
        dc.reset()
        (stepped, completed, reset) = dc.update(1)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.remaining() == 3)
        self.assertTrue(dc.current_seq == [1])
        (stepped, completed, reset) = dc.update(2)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.remaining() == 2)
        self.assertTrue(dc.current_seq == [1, 2])
        (stepped, completed, reset) = dc.update(5)
        self.assertTrue(dc.completed)
        self.assertTrue(dc.remaining() == 0)
        self.assertTrue(dc.current_seq == [1, 2, 5])