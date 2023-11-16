"""Tests for dragnn.python.trainer_lib."""
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from dragnn.python import trainer_lib

class TrainerLibTest(test_util.TensorFlowTestCase):

    def testImmutabilityOfArguments(self):
        if False:
            print('Hello World!')
        'Tests that training schedule generation does not change its arguments.'
        pretrain_steps = [1, 2, 3]
        train_steps = [5, 5, 5]
        trainer_lib.generate_target_per_step_schedule(pretrain_steps, train_steps)
        self.assertEqual(pretrain_steps, [1, 2, 3])
        self.assertEqual(train_steps, [5, 5, 5])

    def testTrainingScheduleGenerationAndDeterminism(self):
        if False:
            return 10
        'Non-trivial schedule, check generation and determinism.'
        pretrain_steps = [1, 2, 3]
        train_steps = [5, 5, 5]
        generated_schedule = trainer_lib.generate_target_per_step_schedule(pretrain_steps, train_steps)
        expected_schedule = [0, 1, 1, 2, 2, 2, 1, 0, 2, 1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
        self.assertEqual(generated_schedule, expected_schedule)

    def testNoPretrainSteps(self):
        if False:
            return 10
        'Edge case, 1 target, no pretrain.'
        generated_schedule = trainer_lib.generate_target_per_step_schedule([0], [10])
        expected_schedule = [0] * 10
        self.assertEqual(generated_schedule, expected_schedule)

    def testNoTrainSteps(self):
        if False:
            return 10
        'Edge case, 1 target, only pretrain.'
        generated_schedule = trainer_lib.generate_target_per_step_schedule([10], [0])
        expected_schedule = [0] * 10
        self.assertEqual(generated_schedule, expected_schedule)
if __name__ == '__main__':
    googletest.main()