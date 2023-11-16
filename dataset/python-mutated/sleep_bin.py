"""Simple binary for sleep."""
import math
import sys
import time
from absl import app
import numpy as np
import tensorflow as tf
from tensorflow.examples.custom_ops_doc.sleep import sleep_op

def stack50(op, delay):
    if False:
        print('Hello World!')
    'Create a tf.stack of 50 sleep ops.\n\n  Args:\n    op: The sleep op, either sleep_op.SyncSleep or sleep_op.AsyncSleep.\n    delay: Each op should finish at least float `delay` seconds after it starts.\n  '
    n = 50
    delays = delay + tf.range(0, n, dtype=float) / 10000.0
    start_t = time.time()
    func = tf.function(lambda : tf.stack([op(delays[i]) for i in range(n)]))
    r_numpy = func().numpy()
    end_t = time.time()
    print('')
    print('Total time = %5.3f seconds using %s' % (end_t - start_t, str(op)))
    print('Returned values from the ops:')
    np.set_printoptions(precision=4, suppress=True)
    print(r_numpy)
    sys.stdout.flush()

def main(argv):
    if False:
        for i in range(10):
            print('nop')
    del argv
    delay_seconds = 1.0
    print('\nUsing synchronous sleep op with each of 50 ops sleeping for about %0.2f seconds,\nso total time is about %0.2f * ceil(50 / NUMBER_OF_THREADS). 16 is a typical\nnumber of threads, which would be %0.2f seconds. The actual time will be\na little greater.\n' % (delay_seconds, delay_seconds, delay_seconds * math.ceil(50.0 / 16.0)))
    stack50(sleep_op.SyncSleep, delay_seconds)
    print('\nUsing asynchronous sleep op with each of 50 ops sleeping only as much as\nnecessary so they finish after at least %0.2f seconds. Time that\nan op spends blocked waiting to finish counts as all or part of its delay.\nThe returned values show how long each ops sleeps or 0 if the op does not\nneed to sleep. The expected total time will be a little greater than\nthe requested delay of %0.2f seconds.\n' % (delay_seconds, delay_seconds))
    stack50(sleep_op.AsyncSleep, delay_seconds)
if __name__ == '__main__':
    app.run(main)