from collections import defaultdict
from hamcrest import assert_that, calling, is_, raises
from deepchecks.core import CheckResult, DatasetKind
from deepchecks.vision.base_checks import ModelOnlyCheck, SingleDatasetCheck, TrainTestCheck

def test_run_base_checks(coco_visiondata_train):
    if False:
        i = 10
        return i + 15
    executions = defaultdict(int)

    class DummyCheck(SingleDatasetCheck):

        def initialize_run(self, context, dataset_kind: DatasetKind):
            if False:
                print('Hello World!')
            executions['initialize_run'] += 1

        def update(self, context, batch, dataset_kind: DatasetKind):
            if False:
                for i in range(10):
                    print('nop')
            executions['update'] += 1

        def compute(self, context, dataset_kind: DatasetKind) -> CheckResult:
            if False:
                print('Hello World!')
            executions['compute'] += 1
            return CheckResult(None)

    class DummyTrainTestCheck(TrainTestCheck):

        def initialize_run(self, context):
            if False:
                while True:
                    i = 10
            executions['initialize_run'] += 1

        def update(self, context, batch, dataset_kind: DatasetKind):
            if False:
                return 10
            executions['update'] += 1

        def compute(self, context) -> CheckResult:
            if False:
                print('Hello World!')
            executions['compute'] += 1
            return CheckResult(None)
    DummyCheck().run(coco_visiondata_train)
    DummyTrainTestCheck().run(coco_visiondata_train, coco_visiondata_train)
    assert_that(executions, is_({'initialize_run': 2, 'compute': 2, 'update': 6}))

def test_base_check_raise_not_implemented():
    if False:
        for i in range(10):
            print('nop')
    context = None
    batch = None
    dataset_kind = DatasetKind.TRAIN
    assert_that(calling(SingleDatasetCheck().update).with_args(context, batch, dataset_kind), raises(NotImplementedError))
    assert_that(calling(TrainTestCheck().update).with_args(context, batch, dataset_kind), raises(NotImplementedError))
    assert_that(calling(SingleDatasetCheck().compute).with_args(context, dataset_kind), raises(NotImplementedError))
    assert_that(calling(TrainTestCheck().compute).with_args(context), raises(NotImplementedError))
    assert_that(calling(ModelOnlyCheck().compute).with_args(context), raises(NotImplementedError))

def test_initialize_run():
    if False:
        for i in range(10):
            print('nop')
    assert_that(SingleDatasetCheck().initialize_run(None, None), is_(None))
    assert_that(TrainTestCheck().initialize_run(None), is_(None))
    assert_that(ModelOnlyCheck().initialize_run(None), is_(None))