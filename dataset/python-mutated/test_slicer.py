import pytest
from pymc.step_methods.slicer import Slice
from tests import sampler_fixtures as sf
from tests.helpers import RVsAssignmentStepsTester, StepMethodTester

class TestSliceUniform(sf.SliceFixture, sf.UniformFixture):
    n_samples = 10000
    tune = 1000
    burn = 0
    chains = 4
    min_n_eff = 5000
    rtol = 0.1
    atol = 0.05

class TestStepSlicer(StepMethodTester):

    @pytest.mark.parametrize('step_fn, draws', [(lambda *_: Slice(), 2000), (lambda *_: Slice(blocked=True), 2000)], ids=str)
    def test_step_continuous(self, step_fn, draws):
        if False:
            i = 10
            return i + 15
        self.step_continuous(step_fn, draws)

class TestRVsAssignmentSlicer(RVsAssignmentStepsTester):

    @pytest.mark.parametrize('step, step_kwargs', [(Slice, {})])
    def test_continuous_steps(self, step, step_kwargs):
        if False:
            print('Hello World!')
        self.continuous_steps(step, step_kwargs)