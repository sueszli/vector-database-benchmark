"""Test ModelFlow imports."""
import adanet.experimental as adanet
import tensorflow.compat.v2 as tf

class ModelFlowTest(tf.test.TestCase):

    def test_public(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNotNone(adanet.controllers.SequentialController)
        self.assertIsNotNone(adanet.keras.EnsembleModel)
        self.assertIsNotNone(adanet.keras.MeanEnsemble)
        self.assertIsNotNone(adanet.keras.WeightedEnsemble)
        self.assertIsNotNone(adanet.keras.ModelSearch)
        self.assertIsNotNone(adanet.phases.AutoEnsemblePhase)
        self.assertIsNotNone(adanet.phases.InputPhase)
        self.assertIsNotNone(adanet.phases.KerasTrainerPhase)
        self.assertIsNotNone(adanet.phases.KerasTunerPhase)
        self.assertIsNotNone(adanet.phases.RepeatPhase)
        self.assertIsNotNone(adanet.schedulers.InProcessScheduler)
        self.assertIsNotNone(adanet.storages.InMemoryStorage)
        self.assertIsNotNone(adanet.work_units.KerasTrainerWorkUnit)
        self.assertIsNotNone(adanet.work_units.KerasTunerWorkUnit)
if __name__ == '__main__':
    tf.test.main()