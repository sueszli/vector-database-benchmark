"""Load binary codes stored as tf.Example in a TFRecord table."""
import tensorflow as tf

def ReadFirstCode(dataset):
    if False:
        for i in range(10):
            print('nop')
    'Read the first example from a binary code RecordIO table.'
    for record in tf.python_io.tf_record_iterator(dataset):
        tf_example = tf.train.Example()
        tf_example.ParseFromString(record)
        break
    return tf_example

def LoadBinaryCode(input_config, batch_size):
    if False:
        i = 10
        return i + 15
    'Load a batch of binary codes from a tf.Example dataset.\n\n  Args:\n    input_config: An InputConfig proto containing the input configuration.\n    batch_size: Output batch size of examples.\n\n  Returns:\n    A batched tensor of binary codes.\n  '
    data = input_config.data
    file_list = [data]
    filename_queue = tf.train.string_input_producer(file_list, capacity=4)
    reader = tf.TFRecordReader()
    (_, values) = reader.read(filename_queue)
    serialized_example = tf.reshape(values, shape=[1])
    serialized_features = {'code_shape': tf.FixedLenFeature([3], dtype=tf.int64), 'code': tf.VarLenFeature(tf.float32)}
    example = tf.parse_example(serialized_example, serialized_features)
    z = example['code_shape']
    code_shape = tf.reshape(tf.cast(z, tf.int32), [3])
    code = tf.reshape(tf.sparse_tensor_to_dense(example['code']), code_shape)
    queue_size = 10
    queue = tf.PaddingFIFOQueue(queue_size + 3 * batch_size, dtypes=[code.dtype], shapes=[[None, None, None]])
    enqueue_op = queue.enqueue([code])
    dequeue_code = queue.dequeue_many(batch_size)
    queue_runner = tf.train.queue_runner.QueueRunner(queue, [enqueue_op])
    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, queue_runner)
    return dequeue_code