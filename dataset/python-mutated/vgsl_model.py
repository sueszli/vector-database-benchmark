"""String network description language to define network layouts."""
from __future__ import print_function
import re
import time
import decoder
import errorcounter as ec
import shapes
import tensorflow as tf
import vgsl_input
import vgslspecs
import tensorflow.contrib.slim as slim
from tensorflow.core.framework import summary_pb2
from tensorflow.python.platform import tf_logging as logging
DECAY_STEPS_FACTOR = 16
DECAY_RATE = pow(0.5, 1.0 / DECAY_STEPS_FACTOR)

def Train(train_dir, model_str, train_data, max_steps, master='', task=0, ps_tasks=0, initial_learning_rate=0.001, final_learning_rate=0.001, learning_rate_halflife=160000, optimizer_type='Adam', num_preprocess_threads=1, reader=None):
    if False:
        return 10
    "Testable trainer with no dependence on FLAGS.\n\n  Args:\n    train_dir: Directory to write checkpoints.\n    model_str: Network specification string.\n    train_data: Training data file pattern.\n    max_steps: Number of training steps to run.\n    master: Name of the TensorFlow master to use.\n    task: Task id of this replica running the training. (0 will be master).\n    ps_tasks: Number of tasks in ps job, or 0 if no ps job.\n    initial_learning_rate: Learing rate at start of training.\n    final_learning_rate: Asymptotic minimum learning rate.\n    learning_rate_halflife: Number of steps over which to halve the difference\n      between initial and final learning rate.\n    optimizer_type: One of 'GradientDescent', 'AdaGrad', 'Momentum', 'Adam'.\n    num_preprocess_threads: Number of input threads.\n    reader: Function that returns an actual reader to read Examples from input\n      files. If None, uses tf.TFRecordReader().\n  "
    if master.startswith('local'):
        device = tf.ReplicaDeviceSetter(ps_tasks)
    else:
        device = '/cpu:0'
    with tf.Graph().as_default():
        with tf.device(device):
            model = InitNetwork(train_data, model_str, 'train', initial_learning_rate, final_learning_rate, learning_rate_halflife, optimizer_type, num_preprocess_threads, reader)
            sv = tf.train.Supervisor(logdir=train_dir, is_chief=task == 0, saver=model.saver, save_summaries_secs=10, save_model_secs=30, recovery_wait_secs=5)
            step = 0
            while step < max_steps:
                try:
                    with sv.managed_session(master) as sess:
                        while step < max_steps:
                            (_, step) = model.TrainAStep(sess)
                            if sv.coord.should_stop():
                                break
                except tf.errors.AbortedError as e:
                    logging.error('Received error:%s', e)
                    continue

def Eval(train_dir, eval_dir, model_str, eval_data, decoder_file, num_steps, graph_def_file=None, eval_interval_secs=0, reader=None):
    if False:
        print('Hello World!')
    'Restores a model from a checkpoint and evaluates it.\n\n  Args:\n    train_dir: Directory to find checkpoints.\n    eval_dir: Directory to write summary events.\n    model_str: Network specification string.\n    eval_data: Evaluation data file pattern.\n    decoder_file: File to read to decode the labels.\n    num_steps: Number of eval steps to run.\n    graph_def_file: File to write graph definition to for freezing.\n    eval_interval_secs: How often to run evaluations, or once if 0.\n    reader: Function that returns an actual reader to read Examples from input\n      files. If None, uses tf.TFRecordReader().\n  Returns:\n    (char error rate, word recall error rate, sequence error rate) as percent.\n  Raises:\n    ValueError: If unimplemented feature is used.\n  '
    decode = None
    if decoder_file:
        decode = decoder.Decoder(decoder_file)
    rates = ec.ErrorRates(label_error=None, word_recall_error=None, word_precision_error=None, sequence_error=None)
    with tf.Graph().as_default():
        model = InitNetwork(eval_data, model_str, 'eval', reader=reader)
        sw = tf.summary.FileWriter(eval_dir)
        while True:
            sess = tf.Session('')
            if graph_def_file is not None:
                if not tf.gfile.Exists(graph_def_file):
                    with tf.gfile.FastGFile(graph_def_file, 'w') as f:
                        f.write(sess.graph.as_graph_def(add_shapes=True).SerializeToString())
            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                step = model.Restore(ckpt.model_checkpoint_path, sess)
                if decode:
                    rates = decode.SoftmaxEval(sess, model, num_steps)
                    _AddRateToSummary('Label error rate', rates.label_error, step, sw)
                    _AddRateToSummary('Word recall error rate', rates.word_recall_error, step, sw)
                    _AddRateToSummary('Word precision error rate', rates.word_precision_error, step, sw)
                    _AddRateToSummary('Sequence error rate', rates.sequence_error, step, sw)
                    sw.flush()
                    print('Error rates=', rates)
                else:
                    raise ValueError('Non-softmax decoder evaluation not implemented!')
            if eval_interval_secs:
                time.sleep(eval_interval_secs)
            else:
                break
    return rates

def InitNetwork(input_pattern, model_spec, mode='eval', initial_learning_rate=5e-05, final_learning_rate=5e-05, halflife=1600000, optimizer_type='Adam', num_preprocess_threads=1, reader=None):
    if False:
        while True:
            i = 10
    "Constructs a python tensor flow model defined by model_spec.\n\n  Args:\n    input_pattern: File pattern of the data in tfrecords of Example.\n    model_spec: Concatenation of input spec, model spec and output spec.\n      See Build below for input/output spec. For model spec, see vgslspecs.py\n    mode: One of 'train', 'eval'\n    initial_learning_rate: Initial learning rate for the network.\n    final_learning_rate: Final learning rate for the network.\n    halflife: Number of steps over which to halve the difference between\n              initial and final learning rate for the network.\n    optimizer_type: One of 'GradientDescent', 'AdaGrad', 'Momentum', 'Adam'.\n    num_preprocess_threads: Number of threads to use for image processing.\n    reader: Function that returns an actual reader to read Examples from input\n      files. If None, uses tf.TFRecordReader().\n    Eval tasks need only specify input_pattern and model_spec.\n\n  Returns:\n    A VGSLImageModel class.\n\n  Raises:\n    ValueError: if the model spec syntax is incorrect.\n  "
    model = VGSLImageModel(mode, model_spec, initial_learning_rate, final_learning_rate, halflife)
    left_bracket = model_spec.find('[')
    right_bracket = model_spec.rfind(']')
    if left_bracket < 0 or right_bracket < 0:
        raise ValueError('Failed to find [] in model spec! ', model_spec)
    input_spec = model_spec[:left_bracket]
    layer_spec = model_spec[left_bracket:right_bracket + 1]
    output_spec = model_spec[right_bracket + 1:]
    model.Build(input_pattern, input_spec, layer_spec, output_spec, optimizer_type, num_preprocess_threads, reader)
    return model

class VGSLImageModel(object):
    """Class that builds a tensor flow model for training or evaluation.
  """

    def __init__(self, mode, model_spec, initial_learning_rate, final_learning_rate, halflife):
        if False:
            return 10
        'Constructs a VGSLImageModel.\n\n    Args:\n      mode:        One of "train", "eval"\n      model_spec:  Full model specification string, for reference only.\n      initial_learning_rate: Initial learning rate for the network.\n      final_learning_rate: Final learning rate for the network.\n      halflife: Number of steps over which to halve the difference between\n                initial and final learning rate for the network.\n    '
        self.model_spec = model_spec
        self.layers = None
        self.mode = mode
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.decay_steps = halflife / DECAY_STEPS_FACTOR
        self.decay_rate = DECAY_RATE
        self.labels = None
        self.sparse_labels = None
        self.truths = None
        self.loss = None
        self.train_op = None
        self.global_step = None
        self.output = None
        self.using_ctc = False
        self.saver = None

    def Build(self, input_pattern, input_spec, model_spec, output_spec, optimizer_type, num_preprocess_threads, reader):
        if False:
            for i in range(10):
                print('nop')
        "Builds the model from the separate input/layers/output spec strings.\n\n    Args:\n      input_pattern: File pattern of the data in tfrecords of TF Example format.\n      input_spec: Specification of the input layer:\n        batchsize,height,width,depth (4 comma-separated integers)\n          Training will run with batches of batchsize images, but runtime can\n          use any batch size.\n          height and/or width can be 0 or -1, indicating variable size,\n          otherwise all images must be the given size.\n          depth must be 1 or 3 to indicate greyscale or color.\n          NOTE 1-d image input, treating the y image dimension as depth, can\n          be achieved using S1(1x0)1,3 as the first op in the model_spec, but\n          the y-size of the input must then be fixed.\n      model_spec: Model definition. See vgslspecs.py\n      output_spec: Output layer definition:\n        O(2|1|0)(l|s|c)n output layer with n classes.\n          2 (heatmap) Output is a 2-d vector map of the input (possibly at\n            different scale).\n          1 (sequence) Output is a 1-d sequence of vector values.\n          0 (value) Output is a 0-d single vector value.\n          l uses a logistic non-linearity on the output, allowing multiple\n            hot elements in any output vector value.\n          s uses a softmax non-linearity, with one-hot output in each value.\n          c uses a softmax with CTC. Can only be used with s (sequence).\n          NOTE Only O1s and O1c are currently supported.\n      optimizer_type: One of 'GradientDescent', 'AdaGrad', 'Momentum', 'Adam'.\n      num_preprocess_threads: Number of threads to use for image processing.\n      reader: Function that returns an actual reader to read Examples from input\n        files. If None, uses tf.TFRecordReader().\n    "
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        shape = _ParseInputSpec(input_spec)
        (out_dims, out_func, num_classes) = _ParseOutputSpec(output_spec)
        self.using_ctc = out_func == 'c'
        (images, heights, widths, labels, sparse, _) = vgsl_input.ImageInput(input_pattern, num_preprocess_threads, shape, self.using_ctc, reader)
        self.labels = labels
        self.sparse_labels = sparse
        self.layers = vgslspecs.VGSLSpecs(widths, heights, self.mode == 'train')
        last_layer = self.layers.Build(images, model_spec)
        self._AddOutputs(last_layer, out_dims, out_func, num_classes)
        if self.mode == 'train':
            self._AddOptimizer(optimizer_type)
        self.saver = tf.train.Saver()

    def TrainAStep(self, sess):
        if False:
            return 10
        'Runs a training step in the session.\n\n    Args:\n      sess: Session in which to train the model.\n    Returns:\n      loss, global_step.\n    '
        (_, loss, step) = sess.run([self.train_op, self.loss, self.global_step])
        return (loss, step)

    def Restore(self, checkpoint_path, sess):
        if False:
            while True:
                i = 10
        'Restores the model from the given checkpoint path into the session.\n\n    Args:\n      checkpoint_path: File pathname of the checkpoint.\n      sess:            Session in which to restore the model.\n    Returns:\n      global_step of the model.\n    '
        self.saver.restore(sess, checkpoint_path)
        return tf.train.global_step(sess, self.global_step)

    def RunAStep(self, sess):
        if False:
            i = 10
            return i + 15
        'Runs a step for eval in the session.\n\n    Args:\n      sess:            Session in which to run the model.\n    Returns:\n      output tensor result, labels tensor result.\n    '
        return sess.run([self.output, self.labels])

    def _AddOutputs(self, prev_layer, out_dims, out_func, num_classes):
        if False:
            return 10
        "Adds the output layer and loss function.\n\n    Args:\n      prev_layer:  Output of last layer of main network.\n      out_dims:    Number of output dimensions, 0, 1 or 2.\n      out_func:    Output non-linearity. 's' or 'c'=softmax, 'l'=logistic.\n      num_classes: Number of outputs/size of last output dimension.\n    "
        height_in = shapes.tensor_dim(prev_layer, dim=1)
        (logits, outputs) = self._AddOutputLayer(prev_layer, out_dims, out_func, num_classes)
        if self.mode == 'train':
            self.loss = self._AddLossFunction(logits, height_in, out_dims, out_func)
            tf.summary.scalar('loss', self.loss)
        elif out_dims == 0:
            self.labels = tf.slice(self.labels, [0, 0], [-1, 1])
            self.labels = tf.reshape(self.labels, [-1])
        logging.info('Final output=%s', outputs)
        logging.info('Labels tensor=%s', self.labels)
        self.output = outputs

    def _AddOutputLayer(self, prev_layer, out_dims, out_func, num_classes):
        if False:
            while True:
                i = 10
        "Add the fully-connected logits and SoftMax/Logistic output Layer.\n\n    Args:\n      prev_layer:  Output of last layer of main network.\n      out_dims:    Number of output dimensions, 0, 1 or 2.\n      out_func:    Output non-linearity. 's' or 'c'=softmax, 'l'=logistic.\n      num_classes: Number of outputs/size of last output dimension.\n\n    Returns:\n      logits:  Pre-softmax/logistic fully-connected output shaped to out_dims.\n      outputs: Post-softmax/logistic shaped to out_dims.\n\n    Raises:\n      ValueError: if syntax is incorrect.\n    "
        batch_in = shapes.tensor_dim(prev_layer, dim=0)
        height_in = shapes.tensor_dim(prev_layer, dim=1)
        width_in = shapes.tensor_dim(prev_layer, dim=2)
        depth_in = shapes.tensor_dim(prev_layer, dim=3)
        if out_dims:
            shaped = tf.reshape(prev_layer, [-1, depth_in])
        else:
            shaped = tf.reshape(prev_layer, [-1, height_in * width_in * depth_in])
        logits = slim.fully_connected(shaped, num_classes, activation_fn=None)
        if out_func == 'l':
            raise ValueError('Logistic not yet supported!')
        else:
            output = tf.nn.softmax(logits)
        if out_dims == 2:
            output_shape = [batch_in, height_in, width_in, num_classes]
        elif out_dims == 1:
            output_shape = [batch_in, height_in * width_in, num_classes]
        else:
            output_shape = [batch_in, num_classes]
        output = tf.reshape(output, output_shape, name='Output')
        logits = tf.reshape(logits, output_shape)
        return (logits, output)

    def _AddLossFunction(self, logits, height_in, out_dims, out_func):
        if False:
            print('Hello World!')
        "Add the appropriate loss function.\n\n    Args:\n      logits:  Pre-softmax/logistic fully-connected output shaped to out_dims.\n      height_in:  Height of logits before going into the softmax layer.\n      out_dims:   Number of output dimensions, 0, 1 or 2.\n      out_func:   Output non-linearity. 's' or 'c'=softmax, 'l'=logistic.\n\n    Returns:\n      loss: That which is to be minimized.\n\n    Raises:\n      ValueError: if logistic is used.\n    "
        if out_func == 'c':
            ctc_input = tf.transpose(logits, [1, 0, 2])
            widths = self.layers.GetLengths(dim=2, factor=height_in)
            cross_entropy = tf.nn.ctc_loss(ctc_input, self.sparse_labels, widths)
        elif out_func == 's':
            if out_dims == 2:
                self.labels = _PadLabels3d(logits, self.labels)
            elif out_dims == 1:
                self.labels = _PadLabels2d(shapes.tensor_dim(logits, dim=1), self.labels)
            else:
                self.labels = tf.slice(self.labels, [0, 0], [-1, 1])
                self.labels = tf.reshape(self.labels, [-1])
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels, name='xent')
        else:
            raise ValueError('Logistic not yet supported!')
        return tf.reduce_sum(cross_entropy)

    def _AddOptimizer(self, optimizer_type):
        if False:
            return 10
        "Adds an optimizer with learning rate decay to minimize self.loss.\n\n    Args:\n      optimizer_type: One of 'GradientDescent', 'AdaGrad', 'Momentum', 'Adam'.\n    Raises:\n      ValueError: if the optimizer type is unrecognized.\n    "
        learn_rate_delta = self.initial_learning_rate - self.final_learning_rate
        learn_rate_dec = tf.add(tf.train.exponential_decay(learn_rate_delta, self.global_step, self.decay_steps, self.decay_rate), self.final_learning_rate)
        if optimizer_type == 'GradientDescent':
            opt = tf.train.GradientDescentOptimizer(learn_rate_dec)
        elif optimizer_type == 'AdaGrad':
            opt = tf.train.AdagradOptimizer(learn_rate_dec)
        elif optimizer_type == 'Momentum':
            opt = tf.train.MomentumOptimizer(learn_rate_dec, momentum=0.9)
        elif optimizer_type == 'Adam':
            opt = tf.train.AdamOptimizer(learning_rate=learn_rate_dec)
        else:
            raise ValueError('Invalid optimizer type: ' + optimizer_type)
        tf.summary.scalar('learn_rate', learn_rate_dec)
        self.train_op = opt.minimize(self.loss, global_step=self.global_step, name='train')

def _PadLabels3d(logits, labels):
    if False:
        while True:
            i = 10
    'Pads or slices 3-d labels to match logits.\n\n  Covers the case of 2-d softmax output, when labels is [batch, height, width]\n  and logits is [batch, height, width, onehot]\n  Args:\n    logits: 4-d Pre-softmax fully-connected output.\n    labels: 3-d, but not necessarily matching in size.\n\n  Returns:\n    labels: Resized by padding or clipping to match logits.\n  '
    logits_shape = shapes.tensor_shape(logits)
    labels_shape = shapes.tensor_shape(labels)
    labels = tf.reshape(labels, [-1, labels_shape[2]])
    labels = _PadLabels2d(logits_shape[2], labels)
    labels = tf.reshape(labels, [labels_shape[0], -1])
    labels = _PadLabels2d(logits_shape[1] * logits_shape[2], labels)
    return tf.reshape(labels, [labels_shape[0], logits_shape[1], logits_shape[2]])

def _PadLabels2d(logits_size, labels):
    if False:
        print('Hello World!')
    'Pads or slices the 2nd dimension of 2-d labels to match logits_size.\n\n  Covers the case of 1-d softmax output, when labels is [batch, seq] and\n  logits is [batch, seq, onehot]\n  Args:\n    logits_size: Tensor returned from tf.shape giving the target size.\n    labels:      2-d, but not necessarily matching in size.\n\n  Returns:\n    labels: Resized by padding or clipping the last dimension to logits_size.\n  '
    pad = logits_size - tf.shape(labels)[1]

    def _PadFn():
        if False:
            while True:
                i = 10
        return tf.pad(labels, [[0, 0], [0, pad]])

    def _SliceFn():
        if False:
            i = 10
            return i + 15
        return tf.slice(labels, [0, 0], [-1, logits_size])
    return tf.cond(tf.greater(pad, 0), _PadFn, _SliceFn)

def _ParseInputSpec(input_spec):
    if False:
        while True:
            i = 10
    'Parses input_spec and returns the numbers obtained therefrom.\n\n  Args:\n    input_spec:  Specification of the input layer. See Build.\n\n  Returns:\n    shape:      ImageShape with the desired shape of the input.\n\n  Raises:\n    ValueError: if syntax is incorrect.\n  '
    pattern = re.compile('(\\d+),(\\d+),(\\d+),(\\d+)')
    m = pattern.match(input_spec)
    if m is None:
        raise ValueError('Failed to parse input spec:' + input_spec)
    batch_size = int(m.group(1))
    y_size = int(m.group(2)) if int(m.group(2)) > 0 else None
    x_size = int(m.group(3)) if int(m.group(3)) > 0 else None
    depth = int(m.group(4))
    if depth not in [1, 3]:
        raise ValueError('Depth must be 1 or 3, had:', depth)
    return vgsl_input.ImageShape(batch_size, y_size, x_size, depth)

def _ParseOutputSpec(output_spec):
    if False:
        i = 10
        return i + 15
    'Parses the output spec.\n\n  Args:\n    output_spec: Output layer definition. See Build.\n\n  Returns:\n    out_dims:     2|1|0 for 2-d, 1-d, 0-d.\n    out_func:     l|s|c for logistic, softmax, softmax+CTC\n    num_classes:  Number of classes in output.\n\n  Raises:\n    ValueError: if syntax is incorrect.\n  '
    pattern = re.compile('(O)(0|1|2)(l|s|c)(\\d+)')
    m = pattern.match(output_spec)
    if m is None:
        raise ValueError('Failed to parse output spec:' + output_spec)
    out_dims = int(m.group(2))
    out_func = m.group(3)
    if out_func == 'c' and out_dims != 1:
        raise ValueError('CTC can only be used with a 1-D sequence!')
    num_classes = int(m.group(4))
    return (out_dims, out_func, num_classes)

def _AddRateToSummary(tag, rate, step, sw):
    if False:
        return 10
    'Adds the given rate to the summary with the given tag.\n\n  Args:\n    tag:   Name for this value.\n    rate:  Value to add to the summary. Perhaps an error rate.\n    step:  Global step of the graph for the x-coordinate of the summary.\n    sw:    Summary writer to which to write the rate value.\n  '
    sw.add_summary(summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=tag, simple_value=rate)]), step)