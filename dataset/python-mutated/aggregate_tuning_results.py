from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'After running tuning, use this script to aggregate the results.\n\nUsage:\n\nOUT_DIR="<my_tuning_dir>"\nbazel run -c opt single_task:aggregate_tuning_results -- \\\n    --alsologtostderr \\\n    --tuning_dir="$OUT_DIR"\n'
import ast
import os
from absl import app
from absl import flags
import tensorflow as tf
FLAGS = flags.FLAGS
flags.DEFINE_string('tuning_dir', '', 'Absolute path where results tuning trial folders are found.')

def main(argv):
    if False:
        while True:
            i = 10
    del argv
    try:
        trial_dirs = tf.gfile.ListDirectory(FLAGS.tuning_dir)
    except tf.errors.NotFoundError:
        print('Tuning directory %s does not exist.' % (FLAGS.tuning_dir,))
        return
    metrics = []
    for trial_dir in trial_dirs:
        tuning_results_file = os.path.join(FLAGS.tuning_dir, trial_dir, 'tuning_results.txt')
        if tf.gfile.Exists(tuning_results_file):
            with tf.gfile.FastGFile(tuning_results_file, 'r') as reader:
                for line in reader:
                    metrics.append(ast.literal_eval(line.replace(': nan,', ': 0.0,')))
    if not metrics:
        print('No trials found.')
        return
    num_trials = [m['num_trials'] for m in metrics]
    assert all((n == num_trials[0] for n in num_trials))
    num_trials = num_trials[0]
    print('Found %d completed trials out of %d' % (len(metrics), num_trials))
    sorted_trials = sorted(metrics, key=lambda m: -m['objective'])
    for (i, metrics) in enumerate(sorted_trials):
        hparams = metrics['hparams']
        keys = sorted(hparams.keys())
        print(str(i).ljust(4) + ': ' + '{0:.2f}'.format(metrics['objective']).ljust(10) + '[' + ','.join(['{}={}'.format(k, hparams[k]).ljust(24) for k in keys]) + ']')
if __name__ == '__main__':
    app.run(main)