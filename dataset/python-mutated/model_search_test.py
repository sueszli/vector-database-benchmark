"""Tests for adanet.experimental.keras.ModelSearch."""
import os
import shutil
import sys
import time
from absl import flags
from absl.testing import parameterized
from adanet.experimental.controllers.sequential_controller import SequentialController
from adanet.experimental.keras import testing_utils
from adanet.experimental.keras.ensemble_model import MeanEnsemble
from adanet.experimental.keras.model_search import ModelSearch
from adanet.experimental.phases.autoensemble_phase import AutoEnsemblePhase
from adanet.experimental.phases.autoensemble_phase import GrowStrategy
from adanet.experimental.phases.autoensemble_phase import MeanEnsembler
from adanet.experimental.phases.input_phase import InputPhase
from adanet.experimental.phases.keras_trainer_phase import KerasTrainerPhase
from adanet.experimental.phases.keras_tuner_phase import KerasTunerPhase
from adanet.experimental.phases.repeat_phase import RepeatPhase
from adanet.experimental.storages.in_memory_storage import InMemoryStorage
from kerastuner import tuners
import tensorflow.compat.v2 as tf

class ModelSearchTest(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(ModelSearchTest, self).setUp()
        flags.FLAGS(sys.argv)
        self.test_subdirectory = os.path.join(flags.FLAGS.test_tmpdir, self.id())
        shutil.rmtree(self.test_subdirectory, ignore_errors=True)
        os.makedirs(self.test_subdirectory)

    def tearDown(self):
        if False:
            print('Hello World!')
        super(ModelSearchTest, self).tearDown()
        shutil.rmtree(self.test_subdirectory, ignore_errors=True)

    def test_phases_end_to_end(self):
        if False:
            while True:
                i = 10
        (train_dataset, test_dataset) = testing_utils.get_holdout_data(train_samples=128, test_samples=64, input_shape=(10,), num_classes=10, random_seed=42)
        train_dataset = train_dataset.batch(32)
        test_dataset = test_dataset.batch(32)
        model1 = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(10)])
        model1.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse', metrics=['mae'])
        model2 = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(10)])
        model2.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse', metrics=['mae'])
        ensemble = MeanEnsemble(submodels=[model1, model2], freeze_submodels=False)
        ensemble.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse', metrics=['mae'])
        controller = SequentialController(phases=[InputPhase(train_dataset, test_dataset), KerasTrainerPhase([model1, model2]), KerasTrainerPhase([ensemble])])
        model_search = ModelSearch(controller)
        model_search.run()
        self.assertIsInstance(model_search.get_best_models(num_models=1)[0], MeanEnsemble)

    def test_tuner_end_to_end(self):
        if False:
            while True:
                i = 10
        (train_dataset, test_dataset) = testing_utils.get_holdout_data(train_samples=128, test_samples=64, input_shape=(10,), num_classes=10, random_seed=42)
        train_dataset = train_dataset.batch(32)
        test_dataset = test_dataset.batch(32)

        def build_model(hp):
            if False:
                return 10
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
            model.add(tf.keras.layers.Dense(10, activation='softmax'))
            model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model
        tuner = tuners.RandomSearch(build_model, objective='val_accuracy', max_trials=3, executions_per_trial=1, directory=self.test_subdirectory, project_name='helloworld_tuner', overwrite=True)
        tuner_phase = KerasTunerPhase(tuner)

        def build_ensemble():
            if False:
                while True:
                    i = 10
            ensemble = MeanEnsemble(submodels=tuner_phase.get_best_models(num_models=2))
            ensemble.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse', metrics=['mae'])
            return [ensemble]
        ensemble_phase = KerasTrainerPhase(build_ensemble)
        input_phase = InputPhase(train_dataset, test_dataset)
        controller = SequentialController(phases=[input_phase, tuner_phase, ensemble_phase])
        model_search = ModelSearch(controller)
        model_search.run()
        self.assertIsInstance(model_search.get_best_models(num_models=1)[0], MeanEnsemble)

    def test_autoensemble_end_to_end(self):
        if False:
            i = 10
            return i + 15
        (train_dataset, test_dataset) = testing_utils.get_holdout_data(train_samples=128, test_samples=64, input_shape=(10,), num_classes=10, random_seed=42)
        train_dataset = train_dataset.batch(32)
        test_dataset = test_dataset.batch(32)

        def build_model(hp):
            if False:
                i = 10
                return i + 15
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
            model.add(tf.keras.layers.Dense(10, activation='softmax'))
            model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model
        autoensemble_storage = InMemoryStorage()
        input_phase = InputPhase(train_dataset, test_dataset)
        repeat_phase = RepeatPhase([lambda : KerasTunerPhase(tuners.RandomSearch(build_model, objective='val_accuracy', max_trials=3, executions_per_trial=1, directory=self.test_subdirectory, project_name='helloworld_' + str(int(time.time())), overwrite=True)), lambda : AutoEnsemblePhase(ensemblers=[MeanEnsembler('sparse_categorical_crossentropy', 'adam', ['accuracy'])], ensemble_strategies=[GrowStrategy()], storage=autoensemble_storage)], repetitions=3)
        controller = SequentialController(phases=[input_phase, repeat_phase])
        model_search = ModelSearch(controller)
        model_search.run()
        self.assertIsInstance(model_search.get_best_models(num_models=1)[0], MeanEnsemble)
if __name__ == '__main__':
    tf.enable_v2_behavior()
    tf.test.main()