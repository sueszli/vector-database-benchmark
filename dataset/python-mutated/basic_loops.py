"""Basic loop for training."""
from tensorflow.python.framework import errors
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['train.basic_train_loop'])
def basic_train_loop(supervisor, train_step_fn, args=None, kwargs=None, master=''):
    if False:
        i = 10
        return i + 15
    'Basic loop to train a model.\n\n  Calls `train_step_fn` in a loop to train a model.  The function is called as:\n\n  ```python\n  train_step_fn(session, *args, **kwargs)\n  ```\n\n  It is passed a `tf.compat.v1.Session` in addition to `args` and `kwargs`.  The\n  function\n  typically runs one training step in the session.\n\n  Args:\n    supervisor: `tf.compat.v1.train.Supervisor` to run the training services.\n    train_step_fn: Callable to execute one training step.  Called repeatedly as\n      `train_step_fn(session, *args **kwargs)`.\n    args: Optional positional arguments passed to `train_step_fn`.\n    kwargs: Optional keyword arguments passed to `train_step_fn`.\n    master: Master to use to create the training session.  Defaults to `""`\n      which causes the session to be created in the local process.\n  '
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    should_retry = True
    while should_retry:
        try:
            should_retry = False
            with supervisor.managed_session(master) as sess:
                while not supervisor.should_stop():
                    train_step_fn(sess, *args, **kwargs)
        except errors.AbortedError:
            should_retry = True