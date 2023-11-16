from snips_nlu.pipeline.processing_unit import ProcessingUnit
from snips_nlu.tests.utils import FixtureTest

class DummyProcessingUnit(ProcessingUnit):
    unit_name = 'dummy_processing_unit'

    def persist(self, path):
        if False:
            while True:
                i = 10
        pass

    @classmethod
    def from_path(cls, path, **shared):
        if False:
            return 10
        return cls(config=None, **shared)

    @property
    def fitted(self):
        if False:
            print('Hello World!')
        return True

class TestProcessingUnit(FixtureTest):

    def test_from_path_with_seed(self):
        if False:
            i = 10
            return i + 15
        max_int = 1000000.0
        seed = 1
        unit_0 = DummyProcessingUnit.from_path(None, random_state=seed)
        int_0 = unit_0.random_state.randint(max_int)
        unit_1 = DummyProcessingUnit.from_path(None, random_state=seed)
        int_1 = unit_1.random_state.randint(max_int)
        self.assertEqual(int_0, int_1)