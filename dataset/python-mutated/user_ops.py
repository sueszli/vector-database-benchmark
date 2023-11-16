"""All user ops."""
from tensorflow.python.user_ops.ops import gen_user_ops as _gen_user_ops
from tensorflow.python.user_ops.ops.gen_user_ops import *
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['user_ops.my_fact'])
def my_fact():
    if False:
        return 10
    'Example of overriding the generated code for an Op.'
    return _gen_user_ops.fact()