"""Data Flow Operations."""
import functools
import hashlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export

def _as_type_list(dtypes):
    if False:
        for i in range(10):
            print('nop')
    'Convert dtypes to a list of types.'
    assert dtypes is not None
    if not (isinstance(dtypes, list) or isinstance(dtypes, tuple)):
        return [dtypes]
    else:
        return list(dtypes)

def _as_shape_list(shapes, dtypes, unknown_dim_allowed=False, unknown_rank_allowed=False):
    if False:
        i = 10
        return i + 15
    'Convert shapes to a list of tuples of int (or None).'
    del dtypes
    if unknown_dim_allowed:
        if not isinstance(shapes, collections_abc.Sequence) or not shapes or any((shape is None or isinstance(shape, int) for shape in shapes)):
            raise ValueError('When providing partial shapes, a list of shapes must be provided.')
    if shapes is None:
        return None
    if isinstance(shapes, tensor_shape.TensorShape):
        shapes = [shapes]
    if not isinstance(shapes, (tuple, list)):
        raise TypeError(f'Shapes must be a TensorShape or a list or tuple of TensorShapes, got {type(shapes)} instead.')
    if all((shape is None or isinstance(shape, int) for shape in shapes)):
        shapes = [shapes]
    shapes = [tensor_shape.as_shape(shape) for shape in shapes]
    if not unknown_dim_allowed:
        if any((not shape.is_fully_defined() for shape in shapes)):
            raise ValueError(f'All shapes must be fully defined: {shapes}')
    if not unknown_rank_allowed:
        if any((shape.dims is None for shape in shapes)):
            raise ValueError(f'All shapes must have a defined rank: {shapes}')
    return shapes

def _as_name_list(names, dtypes):
    if False:
        for i in range(10):
            print('nop')
    if names is None:
        return None
    if not isinstance(names, (list, tuple)):
        names = [names]
    if len(names) != len(dtypes):
        raise ValueError(f'List of names must have the same length as the list of dtypes, received len(names)={len(names)},len(dtypes)={len(dtypes)}')
    return list(names)

def _shape_common(s1, s2):
    if False:
        while True:
            i = 10
    'The greatest lower bound (ordered by specificity) TensorShape.'
    s1 = tensor_shape.TensorShape(s1)
    s2 = tensor_shape.TensorShape(s2)
    if s1.ndims is None or s2.ndims is None or s1.ndims != s2.ndims:
        return tensor_shape.unknown_shape()
    d = [d1 if d1 is not None and d1 == d2 else None for (d1, d2) in zip(s1.as_list(), s2.as_list())]
    return tensor_shape.TensorShape(d)

@tf_export('queue.QueueBase', v1=['queue.QueueBase', 'io.QueueBase', 'QueueBase'])
@deprecation.deprecated_endpoints(['io.QueueBase', 'QueueBase'])
class QueueBase:
    """Base class for queue implementations.

  A queue is a TensorFlow data structure that stores tensors across
  multiple steps, and exposes operations that enqueue and dequeue
  tensors.

  Each queue element is a tuple of one or more tensors, where each
  tuple component has a static dtype, and may have a static shape. The
  queue implementations support versions of enqueue and dequeue that
  handle single elements, versions that support enqueuing and
  dequeuing a batch of elements at once.

  See `tf.queue.FIFOQueue` and
  `tf.queue.RandomShuffleQueue` for concrete
  implementations of this class, and instructions on how to create
  them.
  """

    def __init__(self, dtypes, shapes, names, queue_ref):
        if False:
            return 10
        'Constructs a queue object from a queue reference.\n\n    The two optional lists, `shapes` and `names`, must be of the same length\n    as `dtypes` if provided.  The values at a given index `i` indicate the\n    shape and name to use for the corresponding queue component in `dtypes`.\n\n    Args:\n      dtypes:  A list of types.  The length of dtypes must equal the number\n        of tensors in each element.\n      shapes: Constraints on the shapes of tensors in an element:\n        A list of shape tuples or None. This list is the same length\n        as dtypes.  If the shape of any tensors in the element are constrained,\n        all must be; shapes can be None if the shapes should not be constrained.\n      names: Optional list of names.  If provided, the `enqueue()` and\n        `dequeue()` methods will use dictionaries with these names as keys.\n        Must be None or a list or tuple of the same length as `dtypes`.\n      queue_ref: The queue reference, i.e. the output of the queue op.\n\n    Raises:\n      ValueError: If one of the arguments is invalid.\n    '
        self._dtypes = dtypes
        if shapes is not None:
            if len(shapes) != len(dtypes):
                raise ValueError(f'Queue shapes must have the same length as dtypes, received len(shapes)={len(shapes)}, len(dtypes)={len(dtypes)}')
            self._shapes = [tensor_shape.TensorShape(s) for s in shapes]
        else:
            self._shapes = [tensor_shape.unknown_shape() for _ in self._dtypes]
        if names is not None:
            if len(names) != len(dtypes):
                raise ValueError(f'Queue names must have the same length as dtypes,received len(names)={len(names)},len {len(dtypes)}')
            self._names = names
        else:
            self._names = None
        self._queue_ref = queue_ref
        if isinstance(queue_ref, ops.EagerTensor):
            if context.context().scope_name:
                self._name = context.context().scope_name
            else:
                self._name = 'Empty'
            self._resource_deleter = resource_variable_ops.EagerResourceDeleter(queue_ref, None)
        else:
            self._name = self._queue_ref.op.name.split('/')[-1]

    @staticmethod
    def from_list(index, queues):
        if False:
            for i in range(10):
                print('nop')
        'Create a queue using the queue reference from `queues[index]`.\n\n    Args:\n      index: An integer scalar tensor that determines the input that gets\n        selected.\n      queues: A list of `QueueBase` objects.\n\n    Returns:\n      A `QueueBase` object.\n\n    Raises:\n      TypeError: When `queues` is not a list of `QueueBase` objects,\n        or when the data types of `queues` are not all the same.\n    '
        if not queues or not isinstance(queues, list) or (not all((isinstance(x, QueueBase) for x in queues))):
            raise TypeError('A list of queues expected')
        dtypes = queues[0].dtypes
        if not all((dtypes == q.dtypes for q in queues[1:])):
            raise TypeError('Queues do not have matching component dtypes.')
        names = queues[0].names
        if not all((names == q.names for q in queues[1:])):
            raise TypeError('Queues do not have matching component names.')
        queue_shapes = [q.shapes for q in queues]
        reduced_shapes = [functools.reduce(_shape_common, s) for s in zip(*queue_shapes)]
        queue_refs = array_ops_stack.stack([x.queue_ref for x in queues])
        selected_queue = array_ops.gather(queue_refs, index)
        return QueueBase(dtypes=dtypes, shapes=reduced_shapes, names=names, queue_ref=selected_queue)

    @property
    def queue_ref(self):
        if False:
            print('Hello World!')
        'The underlying queue reference.'
        return self._queue_ref

    @property
    def name(self):
        if False:
            print('Hello World!')
        'The name of the underlying queue.'
        if context.executing_eagerly():
            return self._name
        return self._queue_ref.op.name

    @property
    def dtypes(self):
        if False:
            for i in range(10):
                print('nop')
        'The list of dtypes for each component of a queue element.'
        return self._dtypes

    @property
    def shapes(self):
        if False:
            for i in range(10):
                print('nop')
        'The list of shapes for each component of a queue element.'
        return self._shapes

    @property
    def names(self):
        if False:
            i = 10
            return i + 15
        'The list of names for each component of a queue element.'
        return self._names

    def _check_enqueue_dtypes(self, vals):
        if False:
            return 10
        'Validate and convert `vals` to a list of `Tensor`s.\n\n    The `vals` argument can be a Tensor, a list or tuple of tensors, or a\n    dictionary with tensor values.\n\n    If it is a dictionary, the queue must have been constructed with a\n    `names` attribute and the dictionary keys must match the queue names.\n    If the queue was constructed with a `names` attribute, `vals` must\n    be a dictionary.\n\n    Args:\n      vals: A tensor, a list or tuple of tensors, or a dictionary..\n\n    Returns:\n      A list of `Tensor` objects.\n\n    Raises:\n      ValueError: If `vals` is invalid.\n    '
        if isinstance(vals, dict):
            if not self._names:
                raise ValueError('Queue must have names to enqueue a dictionary')
            if sorted(self._names, key=str) != sorted(vals.keys(), key=str):
                raise ValueError(f'Keys in dictionary to enqueue do not match names of Queue.  Dictionary: {sorted(vals.keys())},Queue: {sorted(self._names)}')
            vals = [vals[k] for k in self._names]
        else:
            if self._names:
                raise ValueError('You must enqueue a dictionary in a Queue with names')
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
        tensors = []
        for (i, (val, dtype)) in enumerate(zip(vals, self._dtypes)):
            tensors.append(ops.convert_to_tensor(val, dtype=dtype, name='component_%d' % i))
        return tensors

    def _scope_vals(self, vals):
        if False:
            i = 10
            return i + 15
        'Return a list of values to pass to `name_scope()`.\n\n    Args:\n      vals: A tensor, a list or tuple of tensors, or a dictionary.\n\n    Returns:\n      The values in vals as a list.\n    '
        if isinstance(vals, (list, tuple)):
            return vals
        elif isinstance(vals, dict):
            return vals.values()
        else:
            return [vals]

    def enqueue(self, vals, name=None):
        if False:
            return 10
        'Enqueues one element to this queue.\n\n    If the queue is full when this operation executes, it will block\n    until the element has been enqueued.\n\n    At runtime, this operation may raise an error if the queue is\n    `tf.QueueBase.close` before or during its execution. If the\n    queue is closed before this operation runs,\n    `tf.errors.CancelledError` will be raised. If this operation is\n    blocked, and either (i) the queue is closed by a close operation\n    with `cancel_pending_enqueues=True`, or (ii) the session is\n    `tf.Session.close`,\n    `tf.errors.CancelledError` will be raised.\n\n    Args:\n      vals: A tensor, a list or tuple of tensors, or a dictionary containing\n        the values to enqueue.\n      name: A name for the operation (optional).\n\n    Returns:\n      The operation that enqueues a new tuple of tensors to the queue.\n    '
        with ops.name_scope(name, '%s_enqueue' % self._name, self._scope_vals(vals)) as scope:
            vals = self._check_enqueue_dtypes(vals)
            for (val, shape) in zip(vals, self._shapes):
                val.get_shape().assert_is_compatible_with(shape)
            if self._queue_ref.dtype == _dtypes.resource:
                return gen_data_flow_ops.queue_enqueue_v2(self._queue_ref, vals, name=scope)
            else:
                return gen_data_flow_ops.queue_enqueue(self._queue_ref, vals, name=scope)

    def enqueue_many(self, vals, name=None):
        if False:
            i = 10
            return i + 15
        'Enqueues zero or more elements to this queue.\n\n    This operation slices each component tensor along the 0th dimension to\n    make multiple queue elements. All of the tensors in `vals` must have the\n    same size in the 0th dimension.\n\n    If the queue is full when this operation executes, it will block\n    until all of the elements have been enqueued.\n\n    At runtime, this operation may raise an error if the queue is\n    `tf.QueueBase.close` before or during its execution. If the\n    queue is closed before this operation runs,\n    `tf.errors.CancelledError` will be raised. If this operation is\n    blocked, and either (i) the queue is closed by a close operation\n    with `cancel_pending_enqueues=True`, or (ii) the session is\n    `tf.Session.close`,\n    `tf.errors.CancelledError` will be raised.\n\n    Args:\n      vals: A tensor, a list or tuple of tensors, or a dictionary\n        from which the queue elements are taken.\n      name: A name for the operation (optional).\n\n    Returns:\n      The operation that enqueues a batch of tuples of tensors to the queue.\n    '
        with ops.name_scope(name, '%s_EnqueueMany' % self._name, self._scope_vals(vals)) as scope:
            vals = self._check_enqueue_dtypes(vals)
            batch_dim = tensor_shape.dimension_value(vals[0].get_shape().with_rank_at_least(1)[0])
            batch_dim = tensor_shape.Dimension(batch_dim)
            for (val, shape) in zip(vals, self._shapes):
                val_batch_dim = tensor_shape.dimension_value(val.get_shape().with_rank_at_least(1)[0])
                val_batch_dim = tensor_shape.Dimension(val_batch_dim)
                batch_dim = batch_dim.merge_with(val_batch_dim)
                val.get_shape()[1:].assert_is_compatible_with(shape)
            return gen_data_flow_ops.queue_enqueue_many_v2(self._queue_ref, vals, name=scope)

    def _dequeue_return_value(self, tensors):
        if False:
            print('Hello World!')
        'Return the value to return from a dequeue op.\n\n    If the queue has names, return a dictionary with the\n    names as keys.  Otherwise return either a single tensor\n    or a list of tensors depending on the length of `tensors`.\n\n    Args:\n      tensors: List of tensors from the dequeue op.\n\n    Returns:\n      A single tensor, a list of tensors, or a dictionary\n      of tensors.\n    '
        if self._names:
            return {n: tensors[i] for (i, n) in enumerate(self._names)}
        elif len(tensors) == 1:
            return tensors[0]
        else:
            return tensors

    def dequeue(self, name=None):
        if False:
            i = 10
            return i + 15
        'Dequeues one element from this queue.\n\n    If the queue is empty when this operation executes, it will block\n    until there is an element to dequeue.\n\n    At runtime, this operation may raise an error if the queue is\n    `tf.QueueBase.close` before or during its execution. If the\n    queue is closed, the queue is empty, and there are no pending\n    enqueue operations that can fulfill this request,\n    `tf.errors.OutOfRangeError` will be raised. If the session is\n    `tf.Session.close`,\n    `tf.errors.CancelledError` will be raised.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      The tuple of tensors that was dequeued.\n    '
        if name is None:
            name = '%s_Dequeue' % self._name
        if self._queue_ref.dtype == _dtypes.resource:
            ret = gen_data_flow_ops.queue_dequeue_v2(self._queue_ref, self._dtypes, name=name)
        else:
            ret = gen_data_flow_ops.queue_dequeue(self._queue_ref, self._dtypes, name=name)
        if not context.executing_eagerly():
            op = ret[0].op
            for (output, shape) in zip(op.values(), self._shapes):
                output.set_shape(shape)
        return self._dequeue_return_value(ret)

    def dequeue_many(self, n, name=None):
        if False:
            while True:
                i = 10
        'Dequeues and concatenates `n` elements from this queue.\n\n    This operation concatenates queue-element component tensors along\n    the 0th dimension to make a single component tensor.  All of the\n    components in the dequeued tuple will have size `n` in the 0th dimension.\n\n    If the queue is closed and there are less than `n` elements left, then an\n    `OutOfRange` exception is raised.\n\n    At runtime, this operation may raise an error if the queue is\n    `tf.QueueBase.close` before or during its execution. If the\n    queue is closed, the queue contains fewer than `n` elements, and\n    there are no pending enqueue operations that can fulfill this\n    request, `tf.errors.OutOfRangeError` will be raised. If the\n    session is `tf.Session.close`,\n    `tf.errors.CancelledError` will be raised.\n\n    Args:\n      n: A scalar `Tensor` containing the number of elements to dequeue.\n      name: A name for the operation (optional).\n\n    Returns:\n      The list of concatenated tensors that was dequeued.\n    '
        if name is None:
            name = '%s_DequeueMany' % self._name
        ret = gen_data_flow_ops.queue_dequeue_many_v2(self._queue_ref, n=n, component_types=self._dtypes, name=name)
        if not context.executing_eagerly():
            op = ret[0].op
            batch_dim = tensor_shape.Dimension(tensor_util.constant_value(op.inputs[1]))
            for (output, shape) in zip(op.values(), self._shapes):
                output.set_shape(tensor_shape.TensorShape([batch_dim]).concatenate(shape))
        return self._dequeue_return_value(ret)

    def dequeue_up_to(self, n, name=None):
        if False:
            print('Hello World!')
        'Dequeues and concatenates `n` elements from this queue.\n\n    **Note** This operation is not supported by all queues.  If a queue does not\n    support DequeueUpTo, then a `tf.errors.UnimplementedError` is raised.\n\n    This operation concatenates queue-element component tensors along\n    the 0th dimension to make a single component tensor. If the queue\n    has not been closed, all of the components in the dequeued tuple\n    will have size `n` in the 0th dimension.\n\n    If the queue is closed and there are more than `0` but fewer than\n    `n` elements remaining, then instead of raising a\n    `tf.errors.OutOfRangeError` like `tf.QueueBase.dequeue_many`,\n    less than `n` elements are returned immediately.  If the queue is\n    closed and there are `0` elements left in the queue, then a\n    `tf.errors.OutOfRangeError` is raised just like in `dequeue_many`.\n    Otherwise the behavior is identical to `dequeue_many`.\n\n    Args:\n      n: A scalar `Tensor` containing the number of elements to dequeue.\n      name: A name for the operation (optional).\n\n    Returns:\n      The tuple of concatenated tensors that was dequeued.\n    '
        if name is None:
            name = '%s_DequeueUpTo' % self._name
        ret = gen_data_flow_ops.queue_dequeue_up_to_v2(self._queue_ref, n=n, component_types=self._dtypes, name=name)
        if not context.executing_eagerly():
            op = ret[0].op
            for (output, shape) in zip(op.values(), self._shapes):
                output.set_shape(tensor_shape.TensorShape([None]).concatenate(shape))
        return self._dequeue_return_value(ret)

    def close(self, cancel_pending_enqueues=False, name=None):
        if False:
            i = 10
            return i + 15
        "Closes this queue.\n\n    This operation signals that no more elements will be enqueued in\n    the given queue. Subsequent `enqueue` and `enqueue_many`\n    operations will fail. Subsequent `dequeue` and `dequeue_many`\n    operations will continue to succeed if sufficient elements remain\n    in the queue. Subsequently dequeue and dequeue_many operations\n    that would otherwise block waiting for more elements (if close\n    hadn't been called) will now fail immediately.\n\n    If `cancel_pending_enqueues` is `True`, all pending requests will also\n    be canceled.\n\n    Args:\n      cancel_pending_enqueues: (Optional.) A boolean, defaulting to\n        `False` (described above).\n      name: A name for the operation (optional).\n\n    Returns:\n      The operation that closes the queue.\n    "
        if name is None:
            name = '%s_Close' % self._name
        if self._queue_ref.dtype == _dtypes.resource:
            return gen_data_flow_ops.queue_close_v2(self._queue_ref, cancel_pending_enqueues=cancel_pending_enqueues, name=name)
        else:
            return gen_data_flow_ops.queue_close(self._queue_ref, cancel_pending_enqueues=cancel_pending_enqueues, name=name)

    def is_closed(self, name=None):
        if False:
            while True:
                i = 10
        'Returns true if queue is closed.\n\n    This operation returns true if the queue is closed and false if the queue\n    is open.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      True if the queue is closed and false if the queue is open.\n    '
        if name is None:
            name = '%s_Is_Closed' % self._name
        if self._queue_ref.dtype == _dtypes.resource:
            return gen_data_flow_ops.queue_is_closed_v2(self._queue_ref, name=name)
        else:
            return gen_data_flow_ops.queue_is_closed_(self._queue_ref, name=name)

    def size(self, name=None):
        if False:
            while True:
                i = 10
        'Compute the number of elements in this queue.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      A scalar tensor containing the number of elements in this queue.\n    '
        if name is None:
            name = '%s_Size' % self._name
        if self._queue_ref.dtype == _dtypes.resource:
            return gen_data_flow_ops.queue_size_v2(self._queue_ref, name=name)
        else:
            return gen_data_flow_ops.queue_size(self._queue_ref, name=name)

def _shared_name(shared_name):
    if False:
        print('Hello World!')
    if context.executing_eagerly():
        return str(ops.uid())
    return shared_name

@tf_export('queue.RandomShuffleQueue', v1=['queue.RandomShuffleQueue', 'io.RandomShuffleQueue', 'RandomShuffleQueue'])
@deprecation.deprecated_endpoints(['io.RandomShuffleQueue', 'RandomShuffleQueue'])
class RandomShuffleQueue(QueueBase):
    """A queue implementation that dequeues elements in a random order.

  See `tf.queue.QueueBase` for a description of the methods on
  this class.
  """

    def __init__(self, capacity, min_after_dequeue, dtypes, shapes=None, names=None, seed=None, shared_name=None, name='random_shuffle_queue'):
        if False:
            for i in range(10):
                print('nop')
        'Create a queue that dequeues elements in a random order.\n\n    A `RandomShuffleQueue` has bounded capacity; supports multiple\n    concurrent producers and consumers; and provides exactly-once\n    delivery.\n\n    A `RandomShuffleQueue` holds a list of up to `capacity`\n    elements. Each element is a fixed-length tuple of tensors whose\n    dtypes are described by `dtypes`, and whose shapes are optionally\n    described by the `shapes` argument.\n\n    If the `shapes` argument is specified, each component of a queue\n    element must have the respective fixed shape. If it is\n    unspecified, different queue elements may have different shapes,\n    but the use of `dequeue_many` is disallowed.\n\n    The `min_after_dequeue` argument allows the caller to specify a\n    minimum number of elements that will remain in the queue after a\n    `dequeue` or `dequeue_many` operation completes, to ensure a\n    minimum level of mixing of elements. This invariant is maintained\n    by blocking those operations until sufficient elements have been\n    enqueued. The `min_after_dequeue` argument is ignored after the\n    queue has been closed.\n\n    Args:\n      capacity: An integer. The upper bound on the number of elements\n        that may be stored in this queue.\n      min_after_dequeue: An integer (described above).\n      dtypes:  A list of `DType` objects. The length of `dtypes` must equal\n        the number of tensors in each queue element.\n      shapes: (Optional.) A list of fully-defined `TensorShape` objects\n        with the same length as `dtypes`, or `None`.\n      names: (Optional.) A list of string naming the components in the queue\n        with the same length as `dtypes`, or `None`.  If specified the dequeue\n        methods return a dictionary with the names as keys.\n      seed: A Python integer. Used to create a random seed. See\n        `tf.compat.v1.set_random_seed`\n        for behavior.\n      shared_name: (Optional.) If non-empty, this queue will be shared under\n        the given name across multiple sessions.\n      name: Optional name for the queue operation.\n    '
        dtypes = _as_type_list(dtypes)
        shapes = _as_shape_list(shapes, dtypes)
        names = _as_name_list(names, dtypes)
        (seed1, seed2) = random_seed.get_seed(seed)
        if seed1 is None and seed2 is None:
            (seed1, seed2) = (0, 0)
        elif seed is None and shared_name is not None:
            string = (str(seed1) + shared_name).encode('utf-8')
            seed2 = int(hashlib.md5(string).hexdigest()[:8], 16) & 2147483647
        queue_ref = gen_data_flow_ops.random_shuffle_queue_v2(component_types=dtypes, shapes=shapes, capacity=capacity, min_after_dequeue=min_after_dequeue, seed=seed1, seed2=seed2, shared_name=_shared_name(shared_name), name=name)
        super(RandomShuffleQueue, self).__init__(dtypes, shapes, names, queue_ref)

@tf_export('queue.FIFOQueue', v1=['queue.FIFOQueue', 'FIFOQueue'])
@deprecation.deprecated_endpoints('FIFOQueue')
class FIFOQueue(QueueBase):
    """A queue implementation that dequeues elements in first-in first-out order.

  See `tf.queue.QueueBase` for a description of the methods on
  this class.
  """

    def __init__(self, capacity, dtypes, shapes=None, names=None, shared_name=None, name='fifo_queue'):
        if False:
            print('Hello World!')
        'Creates a queue that dequeues elements in a first-in first-out order.\n\n    A `FIFOQueue` has bounded capacity; supports multiple concurrent\n    producers and consumers; and provides exactly-once delivery.\n\n    A `FIFOQueue` holds a list of up to `capacity` elements. Each\n    element is a fixed-length tuple of tensors whose dtypes are\n    described by `dtypes`, and whose shapes are optionally described\n    by the `shapes` argument.\n\n    If the `shapes` argument is specified, each component of a queue\n    element must have the respective fixed shape. If it is\n    unspecified, different queue elements may have different shapes,\n    but the use of `dequeue_many` is disallowed.\n\n    Args:\n      capacity: An integer. The upper bound on the number of elements\n        that may be stored in this queue.\n      dtypes:  A list of `DType` objects. The length of `dtypes` must equal\n        the number of tensors in each queue element.\n      shapes: (Optional.) A list of fully-defined `TensorShape` objects\n        with the same length as `dtypes`, or `None`.\n      names: (Optional.) A list of string naming the components in the queue\n        with the same length as `dtypes`, or `None`.  If specified the dequeue\n        methods return a dictionary with the names as keys.\n      shared_name: (Optional.) If non-empty, this queue will be shared under\n        the given name across multiple sessions.\n      name: Optional name for the queue operation.\n    '
        dtypes = _as_type_list(dtypes)
        shapes = _as_shape_list(shapes, dtypes)
        names = _as_name_list(names, dtypes)
        with ops.init_scope(), ops.device('CPU'):
            queue_ref = gen_data_flow_ops.fifo_queue_v2(component_types=dtypes, shapes=shapes, capacity=capacity, shared_name=_shared_name(shared_name), name=name)
        super(FIFOQueue, self).__init__(dtypes, shapes, names, queue_ref)

class GPUCompatibleFIFOQueue(QueueBase):
    """A queue implementation that dequeues elements in first-in first-out order.

  GPUCompatibleFIFOQueue is like FIFOQueue, but the queue resource may be placed
  either on a CPU or on a GPU. It is not cross-device: enqueues and dequeues
  will be colocated with the queue resource. GPUCompatibleFIFOQueue only
  supports enqueue and dequeue at the moment, not enqueue_many or dequeue_many.

  See `tf.queue.QueueBase` for a description of the methods on this class.
  """

    def __init__(self, capacity, dtypes, shapes=None, names=None, shared_name=None, name='fifo_queue'):
        if False:
            while True:
                i = 10
        'Creates a queue that dequeues elements in a first-in first-out order.\n\n    A `FIFOQueue` has bounded capacity; supports multiple concurrent\n    producers and consumers; and provides exactly-once delivery.\n\n    A `FIFOQueue` holds a list of up to `capacity` elements. Each\n    element is a fixed-length tuple of tensors whose dtypes are\n    described by `dtypes`, and whose shapes are optionally described\n    by the `shapes` argument.\n\n    If the `shapes` argument is specified, each component of a queue\n    element must have the respective fixed shape. If it is\n    unspecified, different queue elements may have different shapes,\n    but the use of `dequeue_many` is disallowed.\n\n    Args:\n      capacity: An integer. The upper bound on the number of elements\n        that may be stored in this queue.\n      dtypes:  A list of `DType` objects. The length of `dtypes` must equal\n        the number of tensors in each queue element.\n      shapes: (Optional.) A list of fully-defined `TensorShape` objects\n        with the same length as `dtypes`, or `None`.\n      names: (Optional.) A list of string naming the components in the queue\n        with the same length as `dtypes`, or `None`.  If specified the dequeue\n        methods return a dictionary with the names as keys.\n      shared_name: (Optional.) If non-empty, this queue will be shared under\n        the given name across multiple sessions.\n      name: Optional name for the queue operation.\n    '
        dtypes = _as_type_list(dtypes)
        shapes = _as_shape_list(shapes, dtypes)
        names = _as_name_list(names, dtypes)
        with ops.init_scope():
            queue_ref = gen_data_flow_ops.fifo_queue_v2(component_types=dtypes, shapes=shapes, capacity=capacity, shared_name=_shared_name(shared_name), name=name)
        super(GPUCompatibleFIFOQueue, self).__init__(dtypes, shapes, names, queue_ref)

    def enqueue_many(self, vals, name=None):
        if False:
            while True:
                i = 10
        'enqueue_many is not supported on GPUCompatibleFIFOQueue.'
        raise NotImplementedError('GPUCompatibleFIFOQueue does not support enqueue_many or dequeue_many, only enqueue and dequeue.')

    def dequeue_many(self, n, name=None):
        if False:
            return 10
        'dequeue_many is not supported on GPUCompatibleFIFOQueue.'
        raise NotImplementedError('GPUCompatibleFIFOQueue does not support enqueue_many or dequeue_many, only enqueue and dequeue.')

@tf_export('queue.PaddingFIFOQueue', v1=['queue.PaddingFIFOQueue', 'io.PaddingFIFOQueue', 'PaddingFIFOQueue'])
@deprecation.deprecated_endpoints(['io.PaddingFIFOQueue', 'PaddingFIFOQueue'])
class PaddingFIFOQueue(QueueBase):
    """A FIFOQueue that supports batching variable-sized tensors by padding.

  A `PaddingFIFOQueue` may contain components with dynamic shape, while also
  supporting `dequeue_many`.  See the constructor for more details.

  See `tf.queue.QueueBase` for a description of the methods on
  this class.
  """

    def __init__(self, capacity, dtypes, shapes, names=None, shared_name=None, name='padding_fifo_queue'):
        if False:
            return 10
        "Creates a queue that dequeues elements in a first-in first-out order.\n\n    A `PaddingFIFOQueue` has bounded capacity; supports multiple concurrent\n    producers and consumers; and provides exactly-once delivery.\n\n    A `PaddingFIFOQueue` holds a list of up to `capacity` elements. Each\n    element is a fixed-length tuple of tensors whose dtypes are\n    described by `dtypes`, and whose shapes are described by the `shapes`\n    argument.\n\n    The `shapes` argument must be specified; each component of a queue\n    element must have the respective shape.  Shapes of fixed\n    rank but variable size are allowed by setting any shape dimension to None.\n    In this case, the inputs' shape may vary along the given dimension, and\n    `dequeue_many` will pad the given dimension with zeros up to the maximum\n    shape of all elements in the given batch.\n\n    Args:\n      capacity: An integer. The upper bound on the number of elements\n        that may be stored in this queue.\n      dtypes:  A list of `DType` objects. The length of `dtypes` must equal\n        the number of tensors in each queue element.\n      shapes: A list of `TensorShape` objects, with the same length as\n        `dtypes`.  Any dimension in the `TensorShape` containing value\n        `None` is dynamic and allows values to be enqueued with\n         variable size in that dimension.\n      names: (Optional.) A list of string naming the components in the queue\n        with the same length as `dtypes`, or `None`.  If specified the dequeue\n        methods return a dictionary with the names as keys.\n      shared_name: (Optional.) If non-empty, this queue will be shared under\n        the given name across multiple sessions.\n      name: Optional name for the queue operation.\n\n    Raises:\n      ValueError: If shapes is not a list of shapes, or the lengths of dtypes\n        and shapes do not match, or if names is specified and the lengths of\n        dtypes and names do not match.\n    "
        dtypes = _as_type_list(dtypes)
        shapes = _as_shape_list(shapes, dtypes, unknown_dim_allowed=True)
        names = _as_name_list(names, dtypes)
        if len(dtypes) != len(shapes):
            raise ValueError(f'Shapes must be provided for all components, but received {len(dtypes)} dtypes and {len(shapes)} shapes.')
        queue_ref = gen_data_flow_ops.padding_fifo_queue_v2(component_types=dtypes, shapes=shapes, capacity=capacity, shared_name=_shared_name(shared_name), name=name)
        super(PaddingFIFOQueue, self).__init__(dtypes, shapes, names, queue_ref)

@tf_export('queue.PriorityQueue', v1=['queue.PriorityQueue', 'io.PriorityQueue', 'PriorityQueue'])
@deprecation.deprecated_endpoints(['io.PriorityQueue', 'PriorityQueue'])
class PriorityQueue(QueueBase):
    """A queue implementation that dequeues elements in prioritized order.

  See `tf.queue.QueueBase` for a description of the methods on
  this class.
  """

    def __init__(self, capacity, types, shapes=None, names=None, shared_name=None, name='priority_queue'):
        if False:
            print('Hello World!')
        'Creates a queue that dequeues elements in a first-in first-out order.\n\n    A `PriorityQueue` has bounded capacity; supports multiple concurrent\n    producers and consumers; and provides exactly-once delivery.\n\n    A `PriorityQueue` holds a list of up to `capacity` elements. Each\n    element is a fixed-length tuple of tensors whose dtypes are\n    described by `types`, and whose shapes are optionally described\n    by the `shapes` argument.\n\n    If the `shapes` argument is specified, each component of a queue\n    element must have the respective fixed shape. If it is\n    unspecified, different queue elements may have different shapes,\n    but the use of `dequeue_many` is disallowed.\n\n    Enqueues and Dequeues to the `PriorityQueue` must include an additional\n    tuple entry at the beginning: the `priority`.  The priority must be\n    an int64 scalar (for `enqueue`) or an int64 vector (for `enqueue_many`).\n\n    Args:\n      capacity: An integer. The upper bound on the number of elements\n        that may be stored in this queue.\n      types:  A list of `DType` objects. The length of `types` must equal\n        the number of tensors in each queue element, except the first priority\n        element.  The first tensor in each element is the priority,\n        which must be type int64.\n      shapes: (Optional.) A list of fully-defined `TensorShape` objects,\n        with the same length as `types`, or `None`.\n      names: (Optional.) A list of strings naming the components in the queue\n        with the same length as `dtypes`, or `None`.  If specified, the dequeue\n        methods return a dictionary with the names as keys.\n      shared_name: (Optional.) If non-empty, this queue will be shared under\n        the given name across multiple sessions.\n      name: Optional name for the queue operation.\n    '
        types = _as_type_list(types)
        shapes = _as_shape_list(shapes, types)
        queue_ref = gen_data_flow_ops.priority_queue_v2(component_types=types, shapes=shapes, capacity=capacity, shared_name=_shared_name(shared_name), name=name)
        priority_dtypes = [_dtypes.int64] + types
        priority_shapes = [()] + shapes if shapes else shapes
        super(PriorityQueue, self).__init__(priority_dtypes, priority_shapes, names, queue_ref)

class Barrier:
    """Represents a key-value map that persists across graph executions."""

    def __init__(self, types, shapes=None, shared_name=None, name='barrier'):
        if False:
            for i in range(10):
                print('nop')
        'Creates a barrier that persists across different graph executions.\n\n    A barrier represents a key-value map, where each key is a string, and\n    each value is a tuple of tensors.\n\n    At runtime, the barrier contains \'complete\' and \'incomplete\'\n    elements. A complete element has defined tensors for all\n    components of its value tuple, and may be accessed using\n    take_many. An incomplete element has some undefined components in\n    its value tuple, and may be updated using insert_many.\n\n    The barrier call `take_many` outputs values in a particular order.\n    First, it only outputs completed values.  Second, the order in which\n    completed values are returned matches the order in which their very\n    first component was inserted into the barrier.  So, for example, for this\n    sequence of insertions and removals:\n\n      barrier = Barrier((tf.string, tf.int32), shapes=((), ()))\n      barrier.insert_many(0, keys=["k1", "k2"], values=["a", "b"]).run()\n      barrier.insert_many(1, keys=["k1"], values=[1]).run()\n      barrier.insert_many(0, keys=["k3"], values=["c"]).run()\n      barrier.insert_many(1, keys=["k3"], values=[3]).run()\n      barrier.insert_many(1, keys=["k2"], values=[2]).run()\n\n      (indices, keys, values) = barrier.take_many(2)\n      (indices_val, keys_val, values0_val, values1_val) =\n         session.run([indices, keys, values[0], values[1]])\n\n    The output will be (up to permutation of "k1" and "k2"):\n\n      indices_val == (-2**63, -2**63)\n      keys_val == ("k1", "k2")\n      values0_val == ("a", "b")\n      values1_val == (1, 2)\n\n    Note the key "k2" was inserted into the barrier before "k3".  Even though\n    "k3" was completed first, both are complete by the time\n    take_many is called.  As a result, "k2" is prioritized and "k1" and "k2"\n    are returned first.  "k3" remains in the barrier until the next execution\n    of `take_many`.  Since "k1" and "k2" had their first insertions into\n    the barrier together, their indices are the same (-2**63).  The index\n    of "k3" will be -2**63 + 1, because it was the next new inserted key.\n\n    Args:\n      types: A single dtype or a tuple of dtypes, corresponding to the\n        dtypes of the tensor elements that comprise a value in this barrier.\n      shapes: Optional. Constraints on the shapes of tensors in the values:\n        a single tensor shape tuple; a tuple of tensor shape tuples\n        for each barrier-element tuple component; or None if the shape should\n        not be constrained.\n      shared_name: Optional. If non-empty, this barrier will be shared under\n        the given name across multiple sessions.\n      name: Optional name for the barrier op.\n\n    Raises:\n      ValueError: If one of the `shapes` indicate no elements.\n    '
        self._types = _as_type_list(types)
        if shapes is not None:
            shapes = _as_shape_list(shapes, self._types)
            self._shapes = [tensor_shape.TensorShape(s) for s in shapes]
            for (i, shape) in enumerate(self._shapes):
                if shape.num_elements() == 0:
                    raise ValueError(f"Empty tensors are not supported, but received shape '{shape}' at index {i}")
        else:
            self._shapes = [tensor_shape.unknown_shape() for _ in self._types]
        self._barrier_ref = gen_data_flow_ops.barrier(component_types=self._types, shapes=self._shapes, shared_name=shared_name, name=name)
        if context.executing_eagerly():
            self._name = context.context().scope_name
        else:
            self._name = self._barrier_ref.op.name.split('/')[-1]

    @property
    def barrier_ref(self):
        if False:
            while True:
                i = 10
        'Get the underlying barrier reference.'
        return self._barrier_ref

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        'The name of the underlying barrier.'
        if context.executing_eagerly():
            return self._name
        return self._barrier_ref.op.name

    def insert_many(self, component_index, keys, values, name=None):
        if False:
            return 10
        'For each key, assigns the respective value to the specified component.\n\n    This operation updates each element at component_index.\n\n    Args:\n      component_index: The component of the value that is being assigned.\n      keys: A vector of keys, with length n.\n      values: An any-dimensional tensor of values, which are associated with the\n        respective keys. The first dimension must have length n.\n      name: Optional name for the op.\n\n    Returns:\n      The operation that performs the insertion.\n    Raises:\n      InvalidArgumentsError: If inserting keys and values without elements.\n    '
        if name is None:
            name = '%s_BarrierInsertMany' % self._name
        return gen_data_flow_ops.barrier_insert_many(self._barrier_ref, keys, values, component_index, name=name)

    def take_many(self, num_elements, allow_small_batch=False, timeout=None, name=None):
        if False:
            print('Hello World!')
        'Takes the given number of completed elements from this barrier.\n\n    This operation concatenates completed-element component tensors along\n    the 0th dimension to make a single component tensor.\n\n    If barrier has no completed elements, this operation will block\n    until there are \'num_elements\' elements to take.\n\n    TODO(b/25743580): the semantics of `allow_small_batch` are experimental\n    and may be extended to other cases in the future.\n\n    TODO(ebrevdo): If a take_many(allow_small_batch=True) is blocking\n    already when the barrier is closed, it will block for ever. Fix this\n    by using asynchronous operations.\n\n    Args:\n      num_elements: The number of elements to take.\n      allow_small_batch: If the barrier is closed, don\'t block if there are less\n        completed elements than requested, but instead return all available\n        completed elements.\n      timeout: This specifies the number of milliseconds to block\n        before returning with DEADLINE_EXCEEDED. (This option is not\n        supported yet.)\n      name: A name for the operation (optional).\n\n    Returns:\n      A tuple of (index, key, value_list).\n      "index" is a int64 tensor of length num_elements containing the\n        index of the insert_many call for which the very first component of\n        the given element was inserted into the Barrier, starting with\n        the value -2**63.  Note, this value is different from the\n        index of the insert_many call for which the element was completed.\n      "key" is a string tensor of length num_elements containing the keys.\n      "value_list" is a tuple of tensors, each one with size num_elements\n        in the 0th dimension for each component in the barrier\'s values.\n\n    '
        if name is None:
            name = '%s_BarrierTakeMany' % self._name
        ret = gen_data_flow_ops.barrier_take_many(self._barrier_ref, num_elements, self._types, allow_small_batch, timeout, name=name)
        if not context.executing_eagerly():
            op = ret[0].op
            if allow_small_batch:
                batch_dim = None
            else:
                batch_dim = tensor_shape.Dimension(tensor_util.constant_value(op.inputs[1]))
            op.outputs[0].set_shape(tensor_shape.TensorShape([batch_dim]))
            op.outputs[1].set_shape(tensor_shape.TensorShape([batch_dim]))
            for (output, shape) in zip(op.outputs[2:], self._shapes):
                output.set_shape(tensor_shape.TensorShape([batch_dim]).concatenate(shape))
        return ret

    def close(self, cancel_pending_enqueues=False, name=None):
        if False:
            while True:
                i = 10
        'Closes this barrier.\n\n    This operation signals that no more new key values will be inserted in the\n    given barrier. Subsequent InsertMany operations with new keys will fail.\n    InsertMany operations that just complement already existing keys with other\n    components, will continue to succeed. Subsequent TakeMany operations will\n    continue to succeed if sufficient elements remain in the barrier. Subsequent\n    TakeMany operations that would block will fail immediately.\n\n    If `cancel_pending_enqueues` is `True`, all pending requests to the\n    underlying queue will also be canceled, and completing of already\n    started values is also not acceptable anymore.\n\n    Args:\n      cancel_pending_enqueues: (Optional.) A boolean, defaulting to\n        `False` (described above).\n      name: Optional name for the op.\n\n    Returns:\n      The operation that closes the barrier.\n    '
        if name is None:
            name = '%s_BarrierClose' % self._name
        return gen_data_flow_ops.barrier_close(self._barrier_ref, cancel_pending_enqueues=cancel_pending_enqueues, name=name)

    def ready_size(self, name=None):
        if False:
            i = 10
            return i + 15
        'Compute the number of complete elements in the given barrier.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      A single-element tensor containing the number of complete elements in the\n      given barrier.\n    '
        if name is None:
            name = '%s_BarrierReadySize' % self._name
        return gen_data_flow_ops.barrier_ready_size(self._barrier_ref, name=name)

    def incomplete_size(self, name=None):
        if False:
            return 10
        'Compute the number of incomplete elements in the given barrier.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      A single-element tensor containing the number of incomplete elements in\n      the given barrier.\n    '
        if name is None:
            name = '%s_BarrierIncompleteSize' % self._name
        return gen_data_flow_ops.barrier_incomplete_size(self._barrier_ref, name=name)

@tf_export(v1=['ConditionalAccumulatorBase'])
class ConditionalAccumulatorBase:
    """A conditional accumulator for aggregating gradients.

  Up-to-date gradients (i.e., time step at which gradient was computed is
  equal to the accumulator's time step) are added to the accumulator.

  Extraction of the average gradient is blocked until the required number of
  gradients has been accumulated.
  """

    def __init__(self, dtype, shape, accumulator_ref):
        if False:
            print('Hello World!')
        'Creates a new ConditionalAccumulator.\n\n    Args:\n      dtype: Datatype of the accumulated gradients.\n      shape: Shape of the accumulated gradients.\n      accumulator_ref: A handle to the conditional accumulator, created by sub-\n        classes\n    '
        self._dtype = dtype
        if shape is not None:
            self._shape = tensor_shape.TensorShape(shape)
        else:
            self._shape = tensor_shape.unknown_shape()
        self._accumulator_ref = accumulator_ref
        if context.executing_eagerly():
            self._name = context.context().scope_name
        else:
            self._name = self._accumulator_ref.op.name.split('/')[-1]

    @property
    def accumulator_ref(self):
        if False:
            while True:
                i = 10
        'The underlying accumulator reference.'
        return self._accumulator_ref

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        'The name of the underlying accumulator.'
        return self._name

    @property
    def dtype(self):
        if False:
            print('Hello World!')
        'The datatype of the gradients accumulated by this accumulator.'
        return self._dtype

    def num_accumulated(self, name=None):
        if False:
            while True:
                i = 10
        'Number of gradients that have currently been aggregated in accumulator.\n\n    Args:\n      name: Optional name for the operation.\n\n    Returns:\n      Number of accumulated gradients currently in accumulator.\n    '
        if name is None:
            name = '%s_NumAccumulated' % self._name
        return gen_data_flow_ops.resource_accumulator_num_accumulated(self._accumulator_ref, name=name)

    def set_global_step(self, new_global_step, name=None):
        if False:
            print('Hello World!')
        "Sets the global time step of the accumulator.\n\n    The operation logs a warning if we attempt to set to a time step that is\n    lower than the accumulator's own time step.\n\n    Args:\n      new_global_step: Value of new time step. Can be a variable or a constant\n      name: Optional name for the operation.\n\n    Returns:\n      Operation that sets the accumulator's time step.\n    "
        return gen_data_flow_ops.resource_accumulator_set_global_step(self._accumulator_ref, math_ops.cast(ops.convert_to_tensor(new_global_step), _dtypes.int64), name=name)

@tf_export(v1=['ConditionalAccumulator'])
class ConditionalAccumulator(ConditionalAccumulatorBase):
    """A conditional accumulator for aggregating gradients.

  Up-to-date gradients (i.e., time step at which gradient was computed is
  equal to the accumulator's time step) are added to the accumulator.

  Extraction of the average gradient is blocked until the required number of
  gradients has been accumulated.
  """

    def __init__(self, dtype, shape=None, shared_name=None, name='conditional_accumulator', reduction_type='MEAN'):
        if False:
            while True:
                i = 10
        'Creates a new ConditionalAccumulator.\n\n    Args:\n      dtype: Datatype of the accumulated gradients.\n      shape: Shape of the accumulated gradients.\n      shared_name: Optional. If non-empty, this accumulator will be shared under\n        the given name across multiple sessions.\n      name: Optional name for the accumulator.\n      reduction_type: Reduction type to use when taking the gradient.\n    '
        accumulator_ref = gen_data_flow_ops.resource_conditional_accumulator(dtype=dtype, shape=shape, shared_name=shared_name, name=name, reduction_type=reduction_type)
        if context.executing_eagerly():
            self._resource_deleter = resource_variable_ops.EagerResourceDeleter(handle=accumulator_ref, handle_device=context.context().device_name)
        super(ConditionalAccumulator, self).__init__(dtype, shape, accumulator_ref)

    def apply_grad(self, grad, local_step=0, name=None):
        if False:
            print('Hello World!')
        "Attempts to apply a gradient to the accumulator.\n\n    The attempt is silently dropped if the gradient is stale, i.e., local_step\n    is less than the accumulator's global time step.\n\n    Args:\n      grad: The gradient tensor to be applied.\n      local_step: Time step at which the gradient was computed.\n      name: Optional name for the operation.\n\n    Returns:\n      The operation that (conditionally) applies a gradient to the accumulator.\n\n    Raises:\n      ValueError: If grad is of the wrong shape\n    "
        grad = ops.convert_to_tensor(grad, self._dtype)
        grad.get_shape().assert_is_compatible_with(self._shape)
        local_step = math_ops.cast(ops.convert_to_tensor(local_step), _dtypes.int64)
        return gen_data_flow_ops.resource_accumulator_apply_gradient(self._accumulator_ref, local_step=local_step, gradient=grad, name=name)

    def take_grad(self, num_required, name=None):
        if False:
            print('Hello World!')
        "Attempts to extract the average gradient from the accumulator.\n\n    The operation blocks until sufficient number of gradients have been\n    successfully applied to the accumulator.\n\n    Once successful, the following actions are also triggered:\n\n    - Counter of accumulated gradients is reset to 0.\n    - Aggregated gradient is reset to 0 tensor.\n    - Accumulator's internal time step is incremented by 1.\n\n    Args:\n      num_required: Number of gradients that needs to have been aggregated\n      name: Optional name for the operation\n\n    Returns:\n      A tensor holding the value of the average gradient.\n\n    Raises:\n      InvalidArgumentError: If num_required < 1\n    "
        out = gen_data_flow_ops.resource_accumulator_take_gradient(self._accumulator_ref, num_required, dtype=self._dtype, name=name)
        out.set_shape(self._shape)
        return out

@tf_export(v1=['sparse.SparseConditionalAccumulator', 'SparseConditionalAccumulator'])
class SparseConditionalAccumulator(ConditionalAccumulatorBase):
    """A conditional accumulator for aggregating sparse gradients.

  Sparse gradients are represented by `IndexedSlices`.

  Up-to-date gradients (i.e., time step at which gradient was computed is
  equal to the accumulator's time step) are added to the accumulator.

  Extraction of the average gradient is blocked until the required number of
  gradients has been accumulated.

  Args:
    dtype: Datatype of the accumulated gradients.
    shape: Shape of the accumulated gradients.
    shared_name: Optional. If non-empty, this accumulator will be shared under
      the given name across multiple sessions.
    name: Optional name for the accumulator.
    reduction_type: Reduction type to use when taking the gradient.
  """

    def __init__(self, dtype, shape=None, shared_name=None, name='sparse_conditional_accumulator', reduction_type='MEAN'):
        if False:
            for i in range(10):
                print('nop')
        accumulator_ref = gen_data_flow_ops.sparse_conditional_accumulator(dtype=dtype, shape=shape, shared_name=shared_name, name=name, reduction_type=reduction_type)
        super(SparseConditionalAccumulator, self).__init__(dtype, shape, accumulator_ref)

    def apply_indexed_slices_grad(self, grad, local_step=0, name=None):
        if False:
            print('Hello World!')
        "Attempts to apply a gradient to the accumulator.\n\n    The attempt is silently dropped if the gradient is stale, i.e., `local_step`\n    is less than the accumulator's global time step.\n\n    Args:\n      grad: The gradient `IndexedSlices` to be applied.\n      local_step: Time step at which the gradient was computed.\n      name: Optional name for the operation.\n\n    Returns:\n      The operation that (conditionally) applies a gradient to the accumulator.\n\n    Raises:\n      InvalidArgumentError: If grad is of the wrong shape\n    "
        return self.apply_grad(grad_indices=grad.indices, grad_values=grad.values, grad_shape=grad.dense_shape, local_step=local_step, name=name)

    def apply_grad(self, grad_indices, grad_values, grad_shape=None, local_step=0, name=None):
        if False:
            for i in range(10):
                print('nop')
        "Attempts to apply a sparse gradient to the accumulator.\n\n    The attempt is silently dropped if the gradient is stale, i.e., `local_step`\n    is less than the accumulator's global time step.\n\n    A sparse gradient is represented by its indices, values and possibly empty\n    or None shape. Indices must be a vector representing the locations of\n    non-zero entries in the tensor. Values are the non-zero slices of the\n    gradient, and must have the same first dimension as indices, i.e., the nnz\n    represented by indices and values must be consistent. Shape, if not empty or\n    None, must be consistent with the accumulator's shape (if also provided).\n\n    Example:\n      A tensor [[0, 0], [0, 1], [2, 3]] can be represented\n        indices: [1,2]\n        values: [[0,1],[2,3]]\n        shape: [3, 2]\n\n    Args:\n      grad_indices: Indices of the sparse gradient to be applied.\n      grad_values: Values of the sparse gradient to be applied.\n      grad_shape: Shape of the sparse gradient to be applied.\n      local_step: Time step at which the gradient was computed.\n      name: Optional name for the operation.\n\n    Returns:\n      The operation that (conditionally) applies a gradient to the accumulator.\n\n    Raises:\n      InvalidArgumentError: If grad is of the wrong shape\n    "
        local_step = math_ops.cast(ops.convert_to_tensor(local_step), _dtypes.int64)
        return gen_data_flow_ops.sparse_accumulator_apply_gradient(self._accumulator_ref, local_step=local_step, gradient_indices=math_ops.cast(grad_indices, _dtypes.int64), gradient_values=grad_values, gradient_shape=math_ops.cast([] if grad_shape is None else grad_shape, _dtypes.int64), has_known_shape=grad_shape is not None, name=name)

    def take_grad(self, num_required, name=None):
        if False:
            for i in range(10):
                print('nop')
        "Attempts to extract the average gradient from the accumulator.\n\n    The operation blocks until sufficient number of gradients have been\n    successfully applied to the accumulator.\n\n    Once successful, the following actions are also triggered:\n    - Counter of accumulated gradients is reset to 0.\n    - Aggregated gradient is reset to 0 tensor.\n    - Accumulator's internal time step is incremented by 1.\n\n    Args:\n      num_required: Number of gradients that needs to have been aggregated\n      name: Optional name for the operation\n\n    Returns:\n      A tuple of indices, values, and shape representing the average gradient.\n\n    Raises:\n      InvalidArgumentError: If `num_required` < 1\n    "
        return gen_data_flow_ops.sparse_accumulator_take_gradient(self._accumulator_ref, num_required, dtype=self._dtype, name=name)

    def take_indexed_slices_grad(self, num_required, name=None):
        if False:
            print('Hello World!')
        "Attempts to extract the average gradient from the accumulator.\n\n    The operation blocks until sufficient number of gradients have been\n    successfully applied to the accumulator.\n\n    Once successful, the following actions are also triggered:\n    - Counter of accumulated gradients is reset to 0.\n    - Aggregated gradient is reset to 0 tensor.\n    - Accumulator's internal time step is incremented by 1.\n\n    Args:\n      num_required: Number of gradients that needs to have been aggregated\n      name: Optional name for the operation\n\n    Returns:\n      An `IndexedSlices` holding the value of the average gradient.\n\n    Raises:\n      InvalidArgumentError: If `num_required` < 1\n    "
        return_val = gen_data_flow_ops.sparse_accumulator_take_gradient(self._accumulator_ref, num_required, dtype=self._dtype, name=name)
        return indexed_slices.IndexedSlices(indices=return_val.indices, values=return_val.values, dense_shape=return_val.shape)

    def num_accumulated(self, name=None):
        if False:
            return 10
        'Number of gradients that have currently been aggregated in accumulator.\n\n    Args:\n      name: Optional name for the operation.\n\n    Returns:\n      Number of accumulated gradients currently in accumulator.\n    '
        if name is None:
            name = '%s_NumAccumulated' % self._name
        return gen_data_flow_ops.accumulator_num_accumulated(self._accumulator_ref, name=name)

    def set_global_step(self, new_global_step, name=None):
        if False:
            for i in range(10):
                print('nop')
        "Sets the global time step of the accumulator.\n\n    The operation logs a warning if we attempt to set to a time step that is\n    lower than the accumulator's own time step.\n\n    Args:\n      new_global_step: Value of new time step. Can be a variable or a constant\n      name: Optional name for the operation.\n\n    Returns:\n      Operation that sets the accumulator's time step.\n    "
        return gen_data_flow_ops.accumulator_set_global_step(self._accumulator_ref, math_ops.cast(ops.convert_to_tensor(new_global_step), _dtypes.int64), name=name)

class BaseStagingArea:
    """Base class for Staging Areas."""
    _identifier = 0
    _lock = threading.Lock()

    def __init__(self, dtypes, shapes=None, names=None, shared_name=None, capacity=0, memory_limit=0):
        if False:
            i = 10
            return i + 15
        if shared_name is None:
            self._name = ops.get_default_graph().unique_name(self.__class__.__name__)
        elif isinstance(shared_name, str):
            self._name = shared_name
        else:
            raise ValueError(f'shared_name must be a string, got {shared_name}')
        self._dtypes = dtypes
        if shapes is not None:
            if len(shapes) != len(dtypes):
                raise ValueError('StagingArea shapes must be the same length as dtypes')
            self._shapes = [tensor_shape.TensorShape(s) for s in shapes]
        else:
            self._shapes = [tensor_shape.unknown_shape() for _ in self._dtypes]
        if names is not None:
            if len(names) != len(dtypes):
                raise ValueError('StagingArea names must be the same length as dtypes')
            self._names = names
        else:
            self._names = None
        self._capacity = capacity
        self._memory_limit = memory_limit
        with ops.name_scope('%s_root' % self._name):
            self._coloc_op = control_flow_ops.no_op()

    @property
    def name(self):
        if False:
            while True:
                i = 10
        'The name of the staging area.'
        return self._name

    @property
    def dtypes(self):
        if False:
            while True:
                i = 10
        'The list of dtypes for each component of a staging area element.'
        return self._dtypes

    @property
    def shapes(self):
        if False:
            while True:
                i = 10
        'The list of shapes for each component of a staging area element.'
        return self._shapes

    @property
    def names(self):
        if False:
            i = 10
            return i + 15
        'The list of names for each component of a staging area element.'
        return self._names

    @property
    def capacity(self):
        if False:
            for i in range(10):
                print('nop')
        'The maximum number of elements of this staging area.'
        return self._capacity

    @property
    def memory_limit(self):
        if False:
            return 10
        'The maximum number of bytes of this staging area.'
        return self._memory_limit

    def _check_put_dtypes(self, vals, indices=None):
        if False:
            i = 10
            return i + 15
        'Validate and convert `vals` to a list of `Tensor`s.\n\n    The `vals` argument can be a Tensor, a list or tuple of tensors, or a\n    dictionary with tensor values.\n\n    If `vals` is a list, then the appropriate indices associated with the\n    values must be provided.\n\n    If it is a dictionary, the staging area must have been constructed with a\n    `names` attribute and the dictionary keys must match the staging area names.\n    `indices` will be inferred from the dictionary keys.\n    If the staging area was constructed with a `names` attribute, `vals` must\n    be a dictionary.\n\n    Checks that the dtype and shape of each value matches that\n    of the staging area.\n\n    Args:\n      vals: A tensor, a list or tuple of tensors, or a dictionary.\n\n    Returns:\n      A (tensors, indices) tuple where `tensors` is a list of `Tensor` objects\n      and `indices` is a list of indices associated with the tensors.\n\n    Raises:\n      ValueError: If `vals` or `indices` is invalid.\n    '
        if isinstance(vals, dict):
            if not self._names:
                raise ValueError('Staging areas must have names to enqueue a dictionary')
            if not set(vals.keys()).issubset(self._names):
                raise ValueError(f'Keys in dictionary to put do not match names of staging area. Dictionary: {sorted(vals.keys())}Queue: {sorted(self._names)}')
            (vals, indices, _) = zip(*[(vals[k], i, k) for (i, k) in enumerate(self._names) if k in vals])
        else:
            if self._names:
                raise ValueError('You must enqueue a dictionary in a staging area with names')
            if indices is None:
                raise ValueError('Indices must be supplied when inserting a list of tensors')
            if len(indices) != len(vals):
                raise ValueError(f"Number of indices {len(indices)} doesn't match number of values {len(vals)}")
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
                indices = [0]
        if not len(vals) <= len(self._dtypes):
            raise ValueError(f'Unexpected number of inputs {len(vals)} vs {len(self._dtypes)}')
        tensors = []
        for (val, i) in zip(vals, indices):
            (dtype, shape) = (self._dtypes[i], self._shapes[i])
            if val.dtype != dtype:
                raise ValueError(f'Datatypes do not match. Received val.dtype {str(val.dtype)} and dtype {str(dtype)}')
            val.get_shape().assert_is_compatible_with(shape)
            tensors.append(ops.convert_to_tensor(val, dtype=dtype, name='component_%d' % i))
        return (tensors, indices)

    def _create_device_transfers(self, tensors):
        if False:
            return 10
        "Encode inter-device transfers if the current device\n    is not the same as the Staging Area's device.\n    "
        if not isinstance(tensors, (tuple, list)):
            tensors = [tensors]
        curr_device_scope = control_flow_ops.no_op().device
        if curr_device_scope != self._coloc_op.device:
            tensors = [array_ops.identity(t) for t in tensors]
        return tensors

    def _get_return_value(self, tensors, indices):
        if False:
            return 10
        'Return the value to return from a get op.\n\n    If the staging area has names, return a dictionary with the\n    names as keys.  Otherwise return either a single tensor\n    or a list of tensors depending on the length of `tensors`.\n\n    Args:\n      tensors: List of tensors from the get op.\n      indices: Indices of associated names and shapes\n\n    Returns:\n      A single tensor, a list of tensors, or a dictionary\n      of tensors.\n    '
        tensors = self._create_device_transfers(tensors)
        for (output, i) in zip(tensors, indices):
            output.set_shape(self._shapes[i])
        if self._names:
            return {self._names[i]: t for (t, i) in zip(tensors, indices)}
        return tensors

    def _scope_vals(self, vals):
        if False:
            print('Hello World!')
        'Return a list of values to pass to `name_scope()`.\n\n    Args:\n      vals: A tensor, a list or tuple of tensors, or a dictionary.\n\n    Returns:\n      The values in vals as a list.\n    '
        if isinstance(vals, (list, tuple)):
            return vals
        elif isinstance(vals, dict):
            return vals.values()
        else:
            return [vals]

class StagingArea(BaseStagingArea):
    """Class for staging inputs. No ordering guarantees.

  A `StagingArea` is a TensorFlow data structure that stores tensors across
  multiple steps, and exposes operations that can put and get tensors.

  Each `StagingArea` element is a tuple of one or more tensors, where each
  tuple component has a static dtype, and may have a static shape.

  The capacity of a `StagingArea` may be bounded or unbounded.
  It supports multiple concurrent producers and consumers; and
  provides exactly-once delivery.

  Each element of a `StagingArea` is a fixed-length tuple of tensors whose
  dtypes are described by `dtypes`, and whose shapes are optionally described
  by the `shapes` argument.

  If the `shapes` argument is specified, each component of a staging area
  element must have the respective fixed shape. If it is
  unspecified, different elements may have different shapes,

  It can be configured with a capacity in which case
  put(values) will block until space becomes available.

  Similarly, it can be configured with a memory limit which
  will block put(values) until space is available.
  This is mostly useful for limiting the number of tensors on
  devices such as GPUs.

  All get() and peek() commands block if the requested data
  is not present in the Staging Area.

  """

    def __init__(self, dtypes, shapes=None, names=None, shared_name=None, capacity=0, memory_limit=0):
        if False:
            while True:
                i = 10
        'Constructs a staging area object.\n\n    The two optional lists, `shapes` and `names`, must be of the same length\n    as `dtypes` if provided.  The values at a given index `i` indicate the\n    shape and name to use for the corresponding queue component in `dtypes`.\n\n    The device scope at the time of object creation determines where the\n    storage for the `StagingArea` will reside.  Calls to `put` will incur a copy\n    to this memory space, if necessary.  Tensors returned by `get` will be\n    placed according to the device scope when `get` is called.\n\n    Args:\n      dtypes:  A list of types.  The length of dtypes must equal the number\n        of tensors in each element.\n      shapes: (Optional.) Constraints on the shapes of tensors in an element.\n        A list of shape tuples or None. This list is the same length\n        as dtypes.  If the shape of any tensors in the element are constrained,\n        all must be; shapes can be None if the shapes should not be constrained.\n      names: (Optional.) If provided, the `get()` and\n        `put()` methods will use dictionaries with these names as keys.\n        Must be None or a list or tuple of the same length as `dtypes`.\n      shared_name: (Optional.) A name to be used for the shared object. By\n        passing the same name to two different python objects they will share\n        the underlying staging area. Must be a string.\n      capacity: (Optional.) Maximum number of elements.\n        An integer. If zero, the Staging Area is unbounded\n      memory_limit: (Optional.) Maximum number of bytes of all tensors\n        in the Staging Area.\n        An integer. If zero, the Staging Area is unbounded\n\n    Raises:\n      ValueError: If one of the arguments is invalid.\n    '
        super(StagingArea, self).__init__(dtypes, shapes, names, shared_name, capacity, memory_limit)

    def put(self, values, name=None):
        if False:
            i = 10
            return i + 15
        "Create an op that places a value into the staging area.\n\n    This operation will block if the `StagingArea` has reached\n    its capacity.\n\n    Args:\n      values: A single tensor, a list or tuple of tensors, or a dictionary with\n        tensor values. The number of elements must match the length of the\n        list provided to the dtypes argument when creating the StagingArea.\n      name: A name for the operation (optional).\n\n    Returns:\n        The created op.\n\n    Raises:\n      ValueError: If the number or type of inputs don't match the staging area.\n    "
        with ops.name_scope(name, '%s_put' % self._name, self._scope_vals(values)) as scope:
            if not isinstance(values, (list, tuple, dict)):
                values = [values]
            indices = list(range(len(values)))
            (vals, _) = self._check_put_dtypes(values, indices)
            with ops.colocate_with(self._coloc_op):
                op = gen_data_flow_ops.stage(values=vals, shared_name=self._name, name=scope, capacity=self._capacity, memory_limit=self._memory_limit)
            return op

    def __internal_get(self, get_fn, name):
        if False:
            while True:
                i = 10
        with ops.colocate_with(self._coloc_op):
            ret = get_fn()
        indices = list(range(len(self._dtypes)))
        return self._get_return_value(ret, indices)

    def get(self, name=None):
        if False:
            return 10
        'Gets one element from this staging area.\n\n    If the staging area is empty when this operation executes, it will block\n    until there is an element to dequeue.\n\n    Note that unlike others ops that can block, like the queue Dequeue\n    operations, this can stop other work from happening.  To avoid this, the\n    intended use is for this to be called only when there will be an element\n    already available.  One method for doing this in a training loop would be to\n    run a `put()` call during a warmup session.run call, and then call both\n    `get()` and `put()` in each subsequent step.\n\n    The placement of the returned tensor will be determined by the current\n    device scope when this function is called.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      The tuple of tensors that was gotten.\n    '
        if name is None:
            name = '%s_get' % self._name
        fn = lambda : gen_data_flow_ops.unstage(dtypes=self._dtypes, shared_name=self._name, name=name, capacity=self._capacity, memory_limit=self._memory_limit)
        return self.__internal_get(fn, name)

    def peek(self, index, name=None):
        if False:
            return 10
        'Peeks at an element in the staging area.\n\n    If the staging area is too small to contain the element at\n    the specified index, it will block until enough elements\n    are inserted to complete the operation.\n\n    The placement of the returned tensor will be determined by\n    the current device scope when this function is called.\n\n    Args:\n      index: The index of the tensor within the staging area\n              to look up.\n      name: A name for the operation (optional).\n\n    Returns:\n      The tuple of tensors that was gotten.\n    '
        if name is None:
            name = '%s_peek' % self._name
        fn = lambda : gen_data_flow_ops.stage_peek(index, dtypes=self._dtypes, shared_name=self._name, name=name, capacity=self._capacity, memory_limit=self._memory_limit)
        return self.__internal_get(fn, name)

    def size(self, name=None):
        if False:
            return 10
        'Returns the number of elements in the staging area.\n\n    Args:\n        name: A name for the operation (optional)\n\n    Returns:\n        The created op\n    '
        if name is None:
            name = '%s_size' % self._name
        return gen_data_flow_ops.stage_size(name=name, shared_name=self._name, dtypes=self._dtypes, capacity=self._capacity, memory_limit=self._memory_limit)

    def clear(self, name=None):
        if False:
            print('Hello World!')
        'Clears the staging area.\n\n    Args:\n        name: A name for the operation (optional)\n\n    Returns:\n        The created op\n    '
        if name is None:
            name = '%s_clear' % self._name
        return gen_data_flow_ops.stage_clear(name=name, shared_name=self._name, dtypes=self._dtypes, capacity=self._capacity, memory_limit=self._memory_limit)

class MapStagingArea(BaseStagingArea):
    """A `MapStagingArea` is a TensorFlow data structure that stores tensors
  across multiple steps, and exposes operations that can put and get tensors.

  Each `MapStagingArea` element is a (key, value) pair.
  Only int64 keys are supported, other types should be
  hashed to produce a key.
  Values are a tuple of one or more tensors.
  Each tuple component has a static dtype,
  and may have a static shape.

  The capacity of a `MapStagingArea` may be bounded or unbounded.
  It supports multiple concurrent producers and consumers; and
  provides exactly-once delivery.

  Each value tuple of a `MapStagingArea` is a fixed-length tuple of tensors
  whose
  dtypes are described by `dtypes`, and whose shapes are optionally described
  by the `shapes` argument.

  If the `shapes` argument is specified, each component of a staging area
  element must have the respective fixed shape. If it is
  unspecified, different elements may have different shapes,

  It behaves like an associative container with support for:

   - put(key, values)
   - peek(key)         like dict.get(key)
   - get(key)          like dict.pop(key)
   - get(key=None)     like dict.popitem()
   - size()
   - clear()

  If ordered a tree structure ordered by key will be used and
  get(key=None) will remove (key, value) pairs in increasing key order.
  Otherwise a hashtable

  It can be configured with a capacity in which case
  put(key, values) will block until space becomes available.

  Similarly, it can be configured with a memory limit which
  will block put(key, values) until space is available.
  This is mostly useful for limiting the number of tensors on
  devices such as GPUs.

  All get() and peek() commands block if the requested
  (key, value) pair is not present in the staging area.

  Partial puts are supported and will be placed in an incomplete
  map until such time as all values associated with the key have
  been inserted. Once completed, this (key, value) pair will be
  inserted into the map. Data in the incomplete map
  counts towards the memory limit, but not towards capacity limit.

  Partial gets from the map are also supported.
  This removes the partially requested tensors from the entry,
  but the entry is only removed from the map once all tensors
  associated with it are removed.
  """

    def __init__(self, dtypes, shapes=None, names=None, shared_name=None, ordered=False, capacity=0, memory_limit=0):
        if False:
            while True:
                i = 10
        'Args:\n\n      dtypes:  A list of types.  The length of dtypes must equal the number\n        of tensors in each element.\n      capacity: (Optional.) Maximum number of elements.\n        An integer. If zero, the Staging Area is unbounded\n      memory_limit: (Optional.) Maximum number of bytes of all tensors\n        in the Staging Area (excluding keys).\n        An integer. If zero, the Staging Area is unbounded\n      ordered: (Optional.) If True the underlying data structure\n        is a tree ordered on key. Otherwise assume a hashtable.\n      shapes: (Optional.) Constraints on the shapes of tensors in an element.\n        A list of shape tuples or None. This list is the same length\n        as dtypes.  If the shape of any tensors in the element are constrained,\n        all must be; shapes can be None if the shapes should not be constrained.\n      names: (Optional.) If provided, the `get()` and\n        `put()` methods will use dictionaries with these names as keys.\n        Must be None or a list or tuple of the same length as `dtypes`.\n      shared_name: (Optional.) A name to be used for the shared object. By\n        passing the same name to two different python objects they will share\n        the underlying staging area. Must be a string.\n\n    Raises:\n      ValueError: If one of the arguments is invalid.\n\n    '
        super(MapStagingArea, self).__init__(dtypes, shapes, names, shared_name, capacity, memory_limit)
        self._ordered = ordered
        if ordered:
            self._put_fn = gen_data_flow_ops.ordered_map_stage
            self._pop_fn = gen_data_flow_ops.ordered_map_unstage
            self._popitem_fn = gen_data_flow_ops.ordered_map_unstage_no_key
            self._peek_fn = gen_data_flow_ops.ordered_map_peek
            self._size_fn = gen_data_flow_ops.ordered_map_size
            self._incomplete_size_fn = gen_data_flow_ops.ordered_map_incomplete_size
            self._clear_fn = gen_data_flow_ops.ordered_map_clear
        else:
            self._put_fn = gen_data_flow_ops.map_stage
            self._pop_fn = gen_data_flow_ops.map_unstage
            self._popitem_fn = gen_data_flow_ops.map_unstage_no_key
            self._peek_fn = gen_data_flow_ops.map_peek
            self._size_fn = gen_data_flow_ops.map_size
            self._incomplete_size_fn = gen_data_flow_ops.map_incomplete_size
            self._clear_fn = gen_data_flow_ops.map_clear

    def put(self, key, vals, indices=None, name=None):
        if False:
            print('Hello World!')
        "Create an op that stores the (key, vals) pair in the staging area.\n\n    Incomplete puts are possible, preferably using a dictionary for vals\n    as the appropriate dtypes and shapes can be inferred from the value names\n    dictionary key values. If vals is a list or tuple, indices must\n    also be specified so that the op knows at which element position\n    to perform the insert.\n\n    This operation will block if the capacity or memory limit of this\n    container is reached.\n\n    Args:\n        key: Key associated with the data\n        vals: Tensor (or a dict/tuple of Tensors) to place\n                into the staging area.\n        indices: (Optional) if vals is a tuple/list, this is required.\n        name: A name for the operation (optional)\n\n    Returns:\n        The created op\n\n    Raises:\n        ValueError: If the number or type of inputs don't match the staging\n        area.\n    "
        with ops.name_scope(name, '%s_put' % self._name, self._scope_vals(vals)) as scope:
            (vals, indices) = self._check_put_dtypes(vals, indices)
            with ops.colocate_with(self._coloc_op):
                op = self._put_fn(key, indices, vals, dtypes=self._dtypes, shared_name=self._name, name=scope, capacity=self._capacity, memory_limit=self._memory_limit)
        return op

    def _get_indices_and_dtypes(self, indices=None):
        if False:
            while True:
                i = 10
        if indices is None:
            indices = list(range(len(self._dtypes)))
        if not isinstance(indices, (tuple, list)):
            raise TypeError(f'Invalid indices type {type(indices)}')
        if len(indices) == 0:
            raise ValueError('Empty indices')
        if all((isinstance(i, str) for i in indices)):
            if self._names is None:
                raise ValueError(f'String indices provided {indices}, but this Staging Area was not created with names.')
            try:
                indices = [self._names.index(n) for n in indices]
            except ValueError:
                raise ValueError(f'Named index not in Staging Area names {self._names}')
        elif all((isinstance(i, int) for i in indices)):
            pass
        else:
            raise TypeError(f'Mixed types in indices {indices}. May only be str or int')
        dtypes = [self._dtypes[i] for i in indices]
        return (indices, dtypes)

    def peek(self, key, indices=None, name=None):
        if False:
            print('Hello World!')
        'Peeks at staging area data associated with the key.\n\n    If the key is not in the staging area, it will block\n    until the associated (key, value) is inserted.\n\n    Args:\n        key: Key associated with the required data\n        indices: Partial list of tensors to retrieve (optional).\n                A list of integer or string indices.\n                String indices are only valid if the Staging Area\n                has names associated with it.\n        name: A name for the operation (optional)\n\n    Returns:\n        The created op\n    '
        if name is None:
            name = '%s_pop' % self._name
        (indices, dtypes) = self._get_indices_and_dtypes(indices)
        with ops.colocate_with(self._coloc_op):
            result = self._peek_fn(key, shared_name=self._name, indices=indices, dtypes=dtypes, name=name, capacity=self._capacity, memory_limit=self._memory_limit)
        return self._get_return_value(result, indices)

    def get(self, key=None, indices=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        'If the key is provided, the associated (key, value) is returned from the staging area.\n\n    If the key is not in the staging area, this method will block until\n    the associated (key, value) is inserted.\n    If no key is provided and the staging area is ordered,\n    the (key, value) with the smallest key will be returned.\n    Otherwise, a random (key, value) will be returned.\n\n    If the staging area is empty when this operation executes,\n    it will block until there is an element to dequeue.\n\n    Args:\n        key: Key associated with the required data (Optional)\n        indices: Partial list of tensors to retrieve (optional).\n                A list of integer or string indices.\n                String indices are only valid if the Staging Area\n                has names associated with it.\n        name: A name for the operation (optional)\n\n    Returns:\n        The created op\n    '
        if key is None:
            return self._popitem(indices=indices, name=name)
        else:
            return self._pop(key, indices=indices, name=name)

    def _pop(self, key, indices=None, name=None):
        if False:
            while True:
                i = 10
        'Remove and return the associated (key, value) is returned from the staging area.\n\n    If the key is not in the staging area, this method will block until\n    the associated (key, value) is inserted.\n    Args:\n        key: Key associated with the required data\n        indices: Partial list of tensors to retrieve (optional).\n                A list of integer or string indices.\n                String indices are only valid if the Staging Area\n                has names associated with it.\n        name: A name for the operation (optional)\n\n    Returns:\n        The created op\n    '
        if name is None:
            name = '%s_get' % self._name
        (indices, dtypes) = self._get_indices_and_dtypes(indices)
        with ops.colocate_with(self._coloc_op):
            result = self._pop_fn(key, shared_name=self._name, indices=indices, dtypes=dtypes, name=name, capacity=self._capacity, memory_limit=self._memory_limit)
        return (key, self._get_return_value(result, indices))

    def _popitem(self, indices=None, name=None):
        if False:
            while True:
                i = 10
        'If the staging area is ordered, the (key, value) with the smallest key will be returned.\n\n    Otherwise, a random (key, value) will be returned.\n    If the staging area is empty when this operation executes,\n    it will block until there is an element to dequeue.\n\n    Args:\n        key: Key associated with the required data\n        indices: Partial list of tensors to retrieve (optional).\n                A list of integer or string indices.\n                String indices are only valid if the Staging Area\n                has names associated with it.\n        name: A name for the operation (optional)\n\n    Returns:\n        The created op\n    '
        if name is None:
            name = '%s_get_nokey' % self._name
        (indices, dtypes) = self._get_indices_and_dtypes(indices)
        with ops.colocate_with(self._coloc_op):
            (key, result) = self._popitem_fn(shared_name=self._name, indices=indices, dtypes=dtypes, name=name, capacity=self._capacity, memory_limit=self._memory_limit)
        key = self._create_device_transfers(key)[0]
        result = self._get_return_value(result, indices)
        return (key, result)

    def size(self, name=None):
        if False:
            return 10
        'Returns the number of elements in the staging area.\n\n    Args:\n        name: A name for the operation (optional)\n\n    Returns:\n        The created op\n    '
        if name is None:
            name = '%s_size' % self._name
        return self._size_fn(shared_name=self._name, name=name, dtypes=self._dtypes, capacity=self._capacity, memory_limit=self._memory_limit)

    def incomplete_size(self, name=None):
        if False:
            print('Hello World!')
        'Returns the number of incomplete elements in the staging area.\n\n    Args:\n        name: A name for the operation (optional)\n\n    Returns:\n        The created op\n    '
        if name is None:
            name = '%s_incomplete_size' % self._name
        return self._incomplete_size_fn(shared_name=self._name, name=name, dtypes=self._dtypes, capacity=self._capacity, memory_limit=self._memory_limit)

    def clear(self, name=None):
        if False:
            print('Hello World!')
        'Clears the staging area.\n\n    Args:\n        name: A name for the operation (optional)\n\n    Returns:\n        The created op\n    '
        if name is None:
            name = '%s_clear' % self._name
        return self._clear_fn(shared_name=self._name, name=name, dtypes=self._dtypes, capacity=self._capacity, memory_limit=self._memory_limit)

class RecordInput:
    """RecordInput asynchronously reads and randomly yields TFRecords.

  A RecordInput Op will continuously read a batch of records asynchronously
  into a buffer of some fixed capacity. It can also asynchronously yield
  random records from this buffer.

  It will not start yielding until at least `buffer_size / 2` elements have been
  placed into the buffer so that sufficient randomization can take place.

  The order the files are read will be shifted each epoch by `shift_amount` so
  that the data is presented in a different order every epoch.
  """

    def __init__(self, file_pattern, batch_size=1, buffer_size=1, parallelism=1, shift_ratio=0, seed=0, name=None, batches=None, compression_type=None):
        if False:
            for i in range(10):
                print('nop')
        'Constructs a RecordInput Op.\n\n    Args:\n      file_pattern: File path to the dataset, possibly containing wildcards.\n        All matching files will be iterated over each epoch.\n      batch_size: How many records to return at a time.\n      buffer_size: The maximum number of records the buffer will contain.\n      parallelism: How many reader threads to use for reading from files.\n      shift_ratio: What percentage of the total number files to move the start\n        file forward by each epoch.\n      seed: Specify the random number seed used by generator that randomizes\n        records.\n      name: Optional name for the operation.\n      batches: None by default, creating a single batch op. Otherwise specifies\n        how many batches to create, which are returned as a list when\n        `get_yield_op()` is called. An example use case is to split processing\n        between devices on one computer.\n      compression_type: The type of compression for the file. Currently ZLIB and\n        GZIP are supported. Defaults to none.\n\n    Raises:\n      ValueError: If one of the arguments is invalid.\n    '
        self._batch_size = batch_size
        if batches is not None:
            self._batch_size *= batches
        self._batches = batches
        self._file_pattern = file_pattern
        self._buffer_size = buffer_size
        self._parallelism = parallelism
        self._shift_ratio = shift_ratio
        self._seed = seed
        self._name = name
        self._compression_type = python_io.TFRecordCompressionType.NONE
        if compression_type is not None:
            self._compression_type = compression_type

    def get_yield_op(self):
        if False:
            return 10
        'Adds a node that yields a group of records every time it is executed.\n    If RecordInput `batches` parameter is not None, it yields a list of\n    record batches with the specified `batch_size`.\n    '
        compression_type = python_io.TFRecordOptions.get_compression_type_string(python_io.TFRecordOptions(self._compression_type))
        records = gen_data_flow_ops.record_input(file_pattern=self._file_pattern, file_buffer_size=self._buffer_size, file_parallelism=self._parallelism, file_shuffle_shift_ratio=self._shift_ratio, batch_size=self._batch_size, file_random_seed=self._seed, compression_type=compression_type, name=self._name)
        if self._batches is None:
            return records
        else:
            with ops.name_scope(self._name):
                batch_list = [[] for _ in range(self._batches)]
                records = array_ops.split(records, self._batch_size, 0)
                for (index, protobuf) in enumerate(records):
                    batch_index = index % self._batches
                    batch_list[batch_index].append(array_ops.reshape(protobuf, []))
                return batch_list