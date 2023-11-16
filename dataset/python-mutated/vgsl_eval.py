"""Model eval separate from training."""
from tensorflow import app
from tensorflow.python.platform import flags
import vgsl_model
flags.DEFINE_string('eval_dir', '/tmp/mdir/eval', 'Directory where to write event logs.')
flags.DEFINE_string('graph_def_file', None, 'Output eval graph definition file.')
flags.DEFINE_string('train_dir', '/tmp/mdir', 'Directory where to find training checkpoints.')
flags.DEFINE_string('model_str', '1,150,600,3[S2(4x150)0,2 Ct5,5,16 Mp2,2 Ct5,5,64 Mp3,3([Lrys64 Lbx128][Lbys64 Lbx128][Lfys64 Lbx128])S3(3x0)2,3Lfx128 Lrx128 S0(1x4)0,3 Do Lfx256]O1c134', 'Network description.')
flags.DEFINE_integer('num_steps', 1000, 'Number of steps to run evaluation.')
flags.DEFINE_integer('eval_interval_secs', 60, 'Time interval between eval runs.')
flags.DEFINE_string('eval_data', None, 'Evaluation data filepattern')
flags.DEFINE_string('decoder', None, 'Charset decoder')
FLAGS = flags.FLAGS

def main(argv):
    if False:
        print('Hello World!')
    del argv
    vgsl_model.Eval(FLAGS.train_dir, FLAGS.eval_dir, FLAGS.model_str, FLAGS.eval_data, FLAGS.decoder, FLAGS.num_steps, FLAGS.graph_def_file, FLAGS.eval_interval_secs)
if __name__ == '__main__':
    app.run()