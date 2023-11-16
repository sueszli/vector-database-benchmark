"""Example of using multiplex op in a SavedModel.

multiplex_2_save.py and multiplex_4_load_use.py are programs that use this.

https://www.tensorflow.org/guide/saved_model
https://www.tensorflow.org/api_docs/python/tf/saved_model/save
"""
import tensorflow as tf

def _get_example_tensors():
    if False:
        i = 10
        return i + 15
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
    return (cond, a, b)

def save(multiplex_op, path):
    if False:
        return 10
    'Save a model that contains the given `multiplex_op`.\n\n  Args:\n    multiplex_op: A multiplex Custom Op, e.g. multiplex_4_op.multiplex. This is\n      parameterized so it can also be used to create an "old" model with an\n      older version of the op, e.g. multiplex_2_op.multiplex.\n    path: Directory to save model to.\n  '
    (example_cond, example_a, example_b) = _get_example_tensors()

    class UseMultiplex(tf.Module):

        @tf.function(input_signature=[tf.TensorSpec.from_tensor(example_cond), tf.TensorSpec.from_tensor(example_a), tf.TensorSpec.from_tensor(example_b)])
        def use_multiplex(self, cond, a, b):
            if False:
                print('Hello World!')
            return multiplex_op(cond, a, b)
    model = UseMultiplex()
    tf.saved_model.save(model, path, signatures=model.use_multiplex.get_concrete_function(tf.TensorSpec.from_tensor(example_cond), tf.TensorSpec.from_tensor(example_a), tf.TensorSpec.from_tensor(example_b)))

def load_and_use(path):
    if False:
        print('Hello World!')
    'Load and used a model that was previously created by `save()`.\n\n  Args:\n    path: Directory to load model from, typically the same directory that was\n      used by save().\n\n  Returns:\n    A tensor that is the result of using the multiplex op that is\n    tf.constant([1, 20, 3, 40, 5], dtype=tf.int64).\n  '
    (example_cond, example_a, example_b) = _get_example_tensors()
    restored = tf.saved_model.load(path)
    return restored.use_multiplex(example_cond, example_a, example_b)