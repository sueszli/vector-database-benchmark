"""VariableV1 class."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.util import tf_should_use
from tensorflow.python.util.tf_export import tf_export
_variable_from_proto_fn = None

def set_variable_from_proto_fn(variable_from_proto_fn):
    if False:
        print('Hello World!')
    'Set the variable class that variable proto defs will be converted to.'
    global _variable_from_proto_fn
    _variable_from_proto_fn = variable_from_proto_fn

@tf_export(v1=['is_variable_initialized'])
@tf_should_use.should_use_result
def is_variable_initialized(variable):
    if False:
        i = 10
        return i + 15
    'Tests if a variable has been initialized.\n\n  Args:\n    variable: A `Variable`.\n\n  Returns:\n    Returns a scalar boolean Tensor, `True` if the variable has been\n    initialized, `False` otherwise.\n  '
    return state_ops.is_variable_initialized(variable)

def default_variable_creator(_, **kwds):
    if False:
        i = 10
        return i + 15
    del kwds
    raise NotImplementedError('ref_variable needs to be imported')

@tf_export(v1=['Variable'])
class VariableV1(variables.Variable):
    """See the [Variables Guide](https://tensorflow.org/guide/variables).

  A variable maintains state in the graph across calls to `run()`. You add a
  variable to the graph by constructing an instance of the class `Variable`.

  The `Variable()` constructor requires an initial value for the variable,
  which can be a `Tensor` of any type and shape. The initial value defines the
  type and shape of the variable. After construction, the type and shape of
  the variable are fixed. The value can be changed using one of the assign
  methods.

  If you want to change the shape of a variable later you have to use an
  `assign` Op with `validate_shape=False`.

  Just like any `Tensor`, variables created with `Variable()` can be used as
  inputs for other Ops in the graph. Additionally, all the operators
  overloaded for the `Tensor` class are carried over to variables, so you can
  also add nodes to the graph by just doing arithmetic on variables.

  ```python
  import tensorflow as tf

  # Create a variable.
  w = tf.Variable(<initial-value>, name=<optional-name>)

  # Use the variable in the graph like any Tensor.
  y = tf.matmul(w, ...another variable or tensor...)

  # The overloaded operators are available too.
  z = tf.sigmoid(w + y)

  # Assign a new value to the variable with `assign()` or a related method.
  w.assign(w + 1.0)
  w.assign_add(1.0)
  ```

  When you launch the graph, variables have to be explicitly initialized before
  you can run Ops that use their value. You can initialize a variable by
  running its *initializer op*, restoring the variable from a save file, or
  simply running an `assign` Op that assigns a value to the variable. In fact,
  the variable *initializer op* is just an `assign` Op that assigns the
  variable's initial value to the variable itself.

  ```python
  # Launch the graph in a session.
  with tf.compat.v1.Session() as sess:
      # Run the variable initializer.
      sess.run(w.initializer)
      # ...you now can run ops that use the value of 'w'...
  ```

  The most common initialization pattern is to use the convenience function
  `global_variables_initializer()` to add an Op to the graph that initializes
  all the variables. You then run that Op after launching the graph.

  ```python
  # Add an Op to initialize global variables.
  init_op = tf.compat.v1.global_variables_initializer()

  # Launch the graph in a session.
  with tf.compat.v1.Session() as sess:
      # Run the Op that initializes global variables.
      sess.run(init_op)
      # ...you can now run any Op that uses variable values...
  ```

  If you need to create a variable with an initial value dependent on another
  variable, use the other variable's `initialized_value()`. This ensures that
  variables are initialized in the right order.

  All variables are automatically collected in the graph where they are
  created. By default, the constructor adds the new variable to the graph
  collection `GraphKeys.GLOBAL_VARIABLES`. The convenience function
  `global_variables()` returns the contents of that collection.

  When building a machine learning model it is often convenient to distinguish
  between variables holding the trainable model parameters and other variables
  such as a `global step` variable used to count training steps. To make this
  easier, the variable constructor supports a `trainable=<bool>` parameter. If
  `True`, the new variable is also added to the graph collection
  `GraphKeys.TRAINABLE_VARIABLES`. The convenience function
  `trainable_variables()` returns the contents of this collection. The
  various `Optimizer` classes use this collection as the default list of
  variables to optimize.

  WARNING: tf.Variable objects by default have a non-intuitive memory model. A
  Variable is represented internally as a mutable Tensor which can
  non-deterministically alias other Tensors in a graph. The set of operations
  which consume a Variable and can lead to aliasing is undetermined and can
  change across TensorFlow versions. Avoid writing code which relies on the
  value of a Variable either changing or not changing as other operations
  happen. For example, using Variable objects or simple functions thereof as
  predicates in a `tf.cond` is dangerous and error-prone:

  ```
  v = tf.Variable(True)
  tf.cond(v, lambda: v.assign(False), my_false_fn)  # Note: this is broken.
  ```

  Here, adding `use_resource=True` when constructing the variable will
  fix any nondeterminism issues:
  ```
  v = tf.Variable(True, use_resource=True)
  tf.cond(v, lambda: v.assign(False), my_false_fn)
  ```

  To use the replacement for variables which does
  not have these issues:

  * Add `use_resource=True` when constructing `tf.Variable`;
  * Call `tf.compat.v1.get_variable_scope().set_use_resource(True)` inside a
    `tf.compat.v1.variable_scope` before the `tf.compat.v1.get_variable()` call.
  """

    def __init__(self, initial_value=None, trainable=None, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, expected_shape=None, import_scope=None, constraint=None, use_resource=None, synchronization=variables.VariableSynchronization.AUTO, aggregation=variables.VariableAggregation.NONE, shape=None):
        if False:
            print('Hello World!')
        "Creates a new variable with value `initial_value`.\n\n    The new variable is added to the graph collections listed in `collections`,\n    which defaults to `[GraphKeys.GLOBAL_VARIABLES]`.\n\n    If `trainable` is `True` the variable is also added to the graph collection\n    `GraphKeys.TRAINABLE_VARIABLES`.\n\n    This constructor creates both a `variable` Op and an `assign` Op to set the\n    variable to its initial value.\n\n    Args:\n      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,\n        which is the initial value for the Variable. The initial value must have\n        a shape specified unless `validate_shape` is set to False. Can also be a\n        callable with no argument that returns the initial value when called. In\n        that case, `dtype` must be specified. (Note that initializer functions\n        from init_ops.py must first be bound to a shape before being used here.)\n      trainable: If `True`, also adds the variable to the graph collection\n        `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as the default\n        list of variables to use by the `Optimizer` classes. Defaults to `True`,\n        unless `synchronization` is set to `ON_READ`, in which case it defaults\n        to `False`.\n      collections: List of graph collections keys. The new variable is added to\n        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.\n      validate_shape: If `False`, allows the variable to be initialized with a\n        value of unknown shape. If `True`, the default, the shape of\n        `initial_value` must be known.\n      caching_device: Optional device string describing where the Variable\n        should be cached for reading.  Defaults to the Variable's device. If not\n        `None`, caches on another device.  Typical use is to cache on the device\n        where the Ops using the Variable reside, to deduplicate copying through\n        `Switch` and other conditional statements.\n      name: Optional name for the variable. Defaults to `'Variable'` and gets\n        uniquified automatically.\n      variable_def: `VariableDef` protocol buffer. If not `None`, recreates the\n        Variable object with its contents, referencing the variable's nodes in\n        the graph, which must already exist. The graph is not changed.\n        `variable_def` and the other arguments are mutually exclusive.\n      dtype: If set, initial_value will be converted to the given type. If\n        `None`, either the datatype will be kept (if `initial_value` is a\n        Tensor), or `convert_to_tensor` will decide.\n      expected_shape: A TensorShape. If set, initial_value is expected to have\n        this shape.\n      import_scope: Optional `string`. Name scope to add to the `Variable.` Only\n        used when initializing from protocol buffer.\n      constraint: An optional projection function to be applied to the variable\n        after being updated by an `Optimizer` (e.g. used to implement norm\n        constraints or value constraints for layer weights). The function must\n        take as input the unprojected Tensor representing the value of the\n        variable and return the Tensor for the projected value (which must have\n        the same shape). Constraints are not safe to use when doing asynchronous\n        distributed training.\n      use_resource: whether to use resource variables.\n      synchronization: Indicates when a distributed a variable will be\n        aggregated. Accepted values are constants defined in the class\n        `tf.VariableSynchronization`. By default the synchronization is set to\n        `AUTO` and the current `DistributionStrategy` chooses when to\n        synchronize.\n      aggregation: Indicates how a distributed variable will be aggregated.\n        Accepted values are constants defined in the class\n        `tf.VariableAggregation`.\n      shape: (optional) The shape of this variable. If None, the shape of\n        `initial_value` will be used. When setting this argument to\n        `tf.TensorShape(None)` (representing an unspecified shape), the variable\n        can be assigned with values of different shapes.\n\n    Raises:\n      ValueError: If both `variable_def` and initial_value are specified.\n      ValueError: If the initial value is not specified, or does not have a\n        shape and `validate_shape` is `True`.\n      RuntimeError: If eager execution is enabled.\n    "
    SaveSliceInfo = variables.Variable.SaveSliceInfo

    def initialized_value(self):
        if False:
            return 10
        with ops.init_scope():
            return cond.cond(is_variable_initialized(self), self.read_value, lambda : self.initial_value)

    @staticmethod
    def from_proto(variable_def, import_scope=None):
        if False:
            for i in range(10):
                print('nop')
        return _variable_from_proto_fn(variable_def=variable_def, import_scope=import_scope)

    @classmethod
    def _variable_call(cls, initial_value=None, trainable=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, import_scope=None, constraint=None, synchronization=variables.VariableSynchronization.AUTO, aggregation=variables.VariableAggregation.NONE, shape=None, experimental_enable_variable_lifting=None, expected_shape=None, collections=None, use_resource=None, **kwargs):
        if False:
            while True:
                i = 10
        'VariableV1 class getter. Useful to force the signature.'
        if cls is not VariableV1:
            return None
        previous_getter = lambda **kwargs: default_variable_creator(None, **kwargs)
        for (_, getter) in ops.get_default_graph()._variable_creator_stack:
            previous_getter = variables._make_getter(getter, previous_getter)
        if aggregation is None:
            aggregation = variables.VariableAggregation.NONE
        return previous_getter(initial_value=initial_value, trainable=trainable, validate_shape=validate_shape, caching_device=caching_device, name=name, variable_def=variable_def, dtype=dtype, import_scope=import_scope, constraint=constraint, synchronization=synchronization, aggregation=aggregation, shape=shape, experimental_enable_variable_lifting=experimental_enable_variable_lifting, expected_shape=expected_shape, collections=collections, use_resource=use_resource)
variable_scope.set_variable_v1(VariableV1)