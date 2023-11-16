"""Test Transformer's schedule manager."""
import tensorflow as tf
from official.transformer.utils import schedule

class ScheduleBaseTester(tf.test.TestCase):

    def test_mutual_exclusivity(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            schedule.Manager(train_steps=100, steps_between_evals=100, train_epochs=2, epochs_between_evals=1, default_train_epochs=None, batch_size=2048, max_length=256)

    def test_step_basis(self):
        if False:
            print('Hello World!')
        manager = schedule.Manager(train_steps=1000, steps_between_evals=100, train_epochs=None, epochs_between_evals=None, default_train_epochs=None, batch_size=2048, max_length=256)
        self.assertEqual(manager.single_iteration_train_steps, 100)
        self.assertIsNone(manager.single_iteration_eval_steps)
        self.assertIsNone(manager.repeat_dataset)

    def test_epoch_basis(self):
        if False:
            while True:
                i = 10
        manager = schedule.Manager(train_steps=None, steps_between_evals=None, train_epochs=10, epochs_between_evals=2, default_train_epochs=None, batch_size=2048, max_length=256)
        self.assertIsNone(manager.single_iteration_train_steps)
        self.assertIsNone(manager.single_iteration_eval_steps)
        self.assertEqual(manager.repeat_dataset, 2)

    def test_step_basis_tpu(self):
        if False:
            print('Hello World!')
        manager = schedule.Manager(train_steps=1000, steps_between_evals=100, train_epochs=None, epochs_between_evals=None, default_train_epochs=None, batch_size=2048, max_length=256, use_tpu=True)
        self.assertEqual(manager.single_iteration_train_steps, 100)
        self.assertEqual(manager.single_iteration_eval_steps, 375)
        self.assertIsNone(manager.repeat_dataset)

    def test_epoch_basis_tpu(self):
        if False:
            while True:
                i = 10
        manager = schedule.Manager(train_steps=None, steps_between_evals=None, train_epochs=10, epochs_between_evals=2, default_train_epochs=None, batch_size=2048, max_length=256, use_tpu=True)
        self.assertEqual(manager.single_iteration_train_steps, schedule.NUM_EXAMPLES[tf.estimator.ModeKeys.TRAIN] * 2 // (2048 / 256))
        self.assertEqual(manager.single_iteration_eval_steps, 375)
        self.assertEqual(manager.repeat_dataset, 2)
if __name__ == '__main__':
    tf.test.main()