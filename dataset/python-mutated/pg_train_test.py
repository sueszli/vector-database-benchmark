from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'Tests for pg_train.\n\nThese tests excersize code paths available through configuration options.\nTraining will be run for just a few steps with the goal being to check that\nnothing crashes.\n'
from absl import flags
import tensorflow as tf
from single_task import defaults
from single_task import run
FLAGS = flags.FLAGS

class TrainTest(tf.test.TestCase):

    def RunTrainingSteps(self, config_string, num_steps=10):
        if False:
            print('Hello World!')
        'Run a few training steps with the given config.\n\n    Just check that nothing crashes.\n\n    Args:\n      config_string: Config encoded in a string. See\n          $REPO_PATH/common/config_lib.py\n      num_steps: Number of training steps to run. Defaults to 10.\n    '
        config = defaults.default_config_with_updates(config_string)
        FLAGS.master = ''
        FLAGS.max_npe = num_steps * config.batch_size
        FLAGS.summary_interval = 1
        FLAGS.logdir = tf.test.get_temp_dir()
        FLAGS.config = config_string
        tf.reset_default_graph()
        run.main(None)

    def testVanillaPolicyGradient(self):
        if False:
            i = 10
            return i + 15
        self.RunTrainingSteps('env=c(task="reverse"),agent=c(algorithm="pg"),timestep_limit=90,batch_size=64')

    def testVanillaPolicyGradient_VariableLengthSequences(self):
        if False:
            while True:
                i = 10
        self.RunTrainingSteps('env=c(task="reverse"),agent=c(algorithm="pg",eos_token=False),timestep_limit=90,batch_size=64')

    def testVanillaActorCritic(self):
        if False:
            print('Hello World!')
        self.RunTrainingSteps('env=c(task="reverse"),agent=c(algorithm="pg",ema_baseline_decay=0.0),timestep_limit=90,batch_size=64')

    def testPolicyGradientWithTopK(self):
        if False:
            return 10
        self.RunTrainingSteps('env=c(task="reverse"),agent=c(algorithm="pg",topk_loss_hparam=1.0,topk=10),timestep_limit=90,batch_size=64')

    def testVanillaActorCriticWithTopK(self):
        if False:
            for i in range(10):
                print('nop')
        self.RunTrainingSteps('env=c(task="reverse"),agent=c(algorithm="pg",ema_baseline_decay=0.0,topk_loss_hparam=1.0,topk=10),timestep_limit=90,batch_size=64')

    def testPolicyGradientWithTopK_VariableLengthSequences(self):
        if False:
            while True:
                i = 10
        self.RunTrainingSteps('env=c(task="reverse"),agent=c(algorithm="pg",topk_loss_hparam=1.0,topk=10,eos_token=False),timestep_limit=90,batch_size=64')

    def testPolicyGradientWithImportanceSampling(self):
        if False:
            print('Hello World!')
        self.RunTrainingSteps('env=c(task="reverse"),agent=c(algorithm="pg",alpha=0.5),timestep_limit=90,batch_size=64')
if __name__ == '__main__':
    tf.test.main()