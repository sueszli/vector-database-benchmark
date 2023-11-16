from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'Tests for ga_train.\n\nTests that ga runs for a few generations without crashing.\n'
from absl import flags
import tensorflow as tf
from single_task import defaults
from single_task import run
FLAGS = flags.FLAGS

class GaTest(tf.test.TestCase):

    def RunTrainingSteps(self, config_string, num_steps=10):
        if False:
            return 10
        'Run a few training steps with the given config.\n\n    Just check that nothing crashes.\n\n    Args:\n      config_string: Config encoded in a string. See\n          $REPO_PATH/common/config_lib.py\n      num_steps: Number of training steps to run. Defaults to 10.\n    '
        config = defaults.default_config_with_updates(config_string)
        FLAGS.max_npe = num_steps * config.batch_size
        FLAGS.logdir = tf.test.get_temp_dir()
        FLAGS.config = config_string
        run.main(None)

    def testGeneticAlgorithm(self):
        if False:
            i = 10
            return i + 15
        self.RunTrainingSteps('env=c(task="reverse"),agent=c(algorithm="ga"),timestep_limit=40,batch_size=64')

    def testUniformRandomSearch(self):
        if False:
            for i in range(10):
                print('nop')
        self.RunTrainingSteps('env=c(task="reverse"),agent=c(algorithm="rand"),timestep_limit=40,batch_size=64')
if __name__ == '__main__':
    tf.test.main()