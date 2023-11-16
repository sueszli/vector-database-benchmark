import pytest
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms import scalarize

class ScalarizeTest(BaseTest):
    """
    Tests for the scalarize transform.
    """

    def setUp(self) -> None:
        if False:
            return 10
        self.x = cp.Variable()
        obj1 = cp.Minimize(cp.square(self.x))
        obj2 = cp.Minimize(cp.square(self.x - 1))
        self.objectives = [obj1, obj2]

    def test_weighted_sum(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test weighted sum.\n        '
        weights = [1, 1]
        scalarized = scalarize.weighted_sum(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.5, places=3)
        weights = [1, 0]
        scalarized = scalarize.weighted_sum(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0, places=3)
        weights = [0, 1]
        scalarized = scalarize.weighted_sum(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 1, places=3)

    def test_targets_and_priorities(self) -> None:
        if False:
            i = 10
            return i + 15
        targets = [1, 1]
        priorities = [1, 1]
        scalarized = scalarize.targets_and_priorities(self.objectives, priorities, targets)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.5, places=3)
        targets = [1, 0]
        priorities = [1, 1]
        scalarized = scalarize.targets_and_priorities(self.objectives, priorities, targets)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 1, places=3)
        limits = [1, 0.25]
        targets = [0, 0]
        priorities = [1, 0.0001]
        scalarized = scalarize.targets_and_priorities(self.objectives, priorities, targets, limits, off_target=1e-05)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.5, places=3)
        targets = [-1, 0]
        priorities = [1, 1]
        max_objectives = [cp.Maximize(-obj.args[0]) for obj in self.objectives]
        scalarized = scalarize.targets_and_priorities(max_objectives, priorities, targets, off_target=1e-05)
        assert scalarized.args[0].is_concave()
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 1, places=3)
        limits = [-1, -0.25]
        targets = [0, 0]
        priorities = [1, 0.0001]
        max_objectives = [cp.Maximize(-obj.args[0]) for obj in self.objectives]
        scalarized = scalarize.targets_and_priorities(max_objectives, priorities, targets, limits, off_target=1e-05)
        assert scalarized.args[0].is_concave()
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.5, places=3)

    def test_mixed_convexity(self) -> None:
        if False:
            i = 10
            return i + 15
        obj_1 = self.objectives[0]
        obj_2 = cp.Maximize(-self.objectives[1].args[0])
        objectives = [obj_1, obj_2]
        targets = [1, -1]
        priorities = [1, 1]
        with pytest.raises(ValueError, match='Scalarized objective is neither convex nor concave'):
            scalarize.targets_and_priorities(objectives, priorities, targets)
        priorities = [1, -1]
        limits = [1, -1]
        scalarized = scalarize.targets_and_priorities(objectives, priorities, targets, limits)
        assert scalarized.args[0].is_convex()
        priorities = [-1, 1]
        limits = [1, -1]
        scalarized = scalarize.targets_and_priorities(objectives, priorities, targets, limits)
        assert scalarized.args[0].is_concave()

    def test_targets_and_priorities_exceptions(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        targets = [1, 1]
        priorities = [1]
        with pytest.raises(AssertionError, match='Number of objectives and priorities'):
            scalarize.targets_and_priorities(self.objectives, priorities, targets)
        priorities = [1, 1]
        targets = [1]
        with pytest.raises(AssertionError, match='Number of objectives and targets'):
            scalarize.targets_and_priorities(self.objectives, priorities, targets)
        priorities = [1, 1]
        targets = [1, 1]
        limits = [1]
        with pytest.raises(AssertionError, match='Number of objectives and limits'):
            scalarize.targets_and_priorities(self.objectives, priorities, targets, limits)
        limits = [1, 1]
        off_target = -1
        with pytest.raises(AssertionError, match='The off_target argument must be nonnegative'):
            scalarize.targets_and_priorities(self.objectives, priorities, targets, limits, off_target)

    def test_max(self) -> None:
        if False:
            return 10
        weights = [1, 2]
        scalarized = scalarize.max(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.5858, places=3)
        weights = [2, 1]
        scalarized = scalarize.max(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.4142, places=3)

    def test_log_sum_exp(self) -> None:
        if False:
            while True:
                i = 10
        weights = [1, 2]
        scalarized = scalarize.log_sum_exp(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.6354, places=3)
        weights = [2, 1]
        scalarized = scalarize.log_sum_exp(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.3646, places=3)