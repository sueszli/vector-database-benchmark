from keras.layers import Lambda, concatenate
from keras.models import Model
import tensorflow as tf

def multi_gpu_model(model, gpus):
    if False:
        return 10
    if isinstance(gpus, (list, tuple)):
        num_gpus = len(gpus)
        target_gpu_ids = gpus
    else:
        num_gpus = gpus
        target_gpu_ids = range(num_gpus)

    def get_slice(data, i, parts):
        if False:
            i = 10
            return i + 15
        shape = tf.shape(data)
        batch_size = shape[:1]
        input_shape = shape[1:]
        step = batch_size // parts
        if i == num_gpus - 1:
            size = batch_size - step * i
        else:
            size = step
        size = tf.concat([size, input_shape], axis=0)
        stride = tf.concat([step, input_shape * 0], axis=0)
        start = stride * i
        return tf.slice(data, start, size)
    all_outputs = []
    for i in range(len(model.outputs)):
        all_outputs.append([])
    for (i, gpu_id) in enumerate(target_gpu_ids):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('replica_%d' % gpu_id):
                inputs = []
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_i = Lambda(get_slice, output_shape=input_shape, arguments={'i': i, 'parts': num_gpus})(x)
                    inputs.append(slice_i)
                outputs = model(inputs)
                if not isinstance(outputs, list):
                    outputs = [outputs]
                for o in range(len(outputs)):
                    all_outputs[o].append(outputs[o])
    with tf.device('/cpu:0'):
        merged = []
        for (name, outputs) in zip(model.output_names, all_outputs):
            merged.append(concatenate(outputs, axis=0, name=name))
        return Model(model.inputs, merged)