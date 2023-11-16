"""Provides functions to prefetch tensors to feed into models."""
import tensorflow as tf

def prefetch(tensor_dict, capacity):
    if False:
        return 10
    "Creates a prefetch queue for tensors.\n\n  Creates a FIFO queue to asynchronously enqueue tensor_dicts and returns a\n  dequeue op that evaluates to a tensor_dict. This function is useful in\n  prefetching preprocessed tensors so that the data is readily available for\n  consumers.\n\n  Example input pipeline when you don't need batching:\n  ----------------------------------------------------\n  key, string_tensor = slim.parallel_reader.parallel_read(...)\n  tensor_dict = decoder.decode(string_tensor)\n  tensor_dict = preprocessor.preprocess(tensor_dict, ...)\n  prefetch_queue = prefetcher.prefetch(tensor_dict, capacity=20)\n  tensor_dict = prefetch_queue.dequeue()\n  outputs = Model(tensor_dict)\n  ...\n  ----------------------------------------------------\n\n  For input pipelines with batching, refer to core/batcher.py\n\n  Args:\n    tensor_dict: a dictionary of tensors to prefetch.\n    capacity: the size of the prefetch queue.\n\n  Returns:\n    a FIFO prefetcher queue\n  "
    names = list(tensor_dict.keys())
    dtypes = [t.dtype for t in tensor_dict.values()]
    shapes = [t.get_shape() for t in tensor_dict.values()]
    prefetch_queue = tf.PaddingFIFOQueue(capacity, dtypes=dtypes, shapes=shapes, names=names, name='prefetch_queue')
    enqueue_op = prefetch_queue.enqueue(tensor_dict)
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(prefetch_queue, [enqueue_op]))
    tf.summary.scalar('queue/%s/fraction_of_%d_full' % (prefetch_queue.name, capacity), tf.cast(prefetch_queue.size(), dtype=tf.float32) * (1.0 / capacity))
    return prefetch_queue