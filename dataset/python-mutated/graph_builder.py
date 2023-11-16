"""Builds parser models."""
import tensorflow as tf
import syntaxnet.load_parser_ops
from tensorflow.python.ops import control_flow_ops as cf
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as tf_saver
from syntaxnet.ops import gen_parser_ops

def BatchedSparseToDense(sparse_indices, output_size):
    if False:
        i = 10
        return i + 15
    'Batch compatible sparse to dense conversion.\n\n  This is useful for one-hot coded target labels.\n\n  Args:\n    sparse_indices: [batch_size] tensor containing one index per batch\n    output_size: needed in order to generate the correct dense output\n\n  Returns:\n    A [batch_size, output_size] dense tensor.\n  '
    eye = tf.diag(tf.fill([output_size], tf.constant(1, tf.float32)))
    return tf.nn.embedding_lookup(eye, sparse_indices)

def EmbeddingLookupFeatures(params, sparse_features, allow_weights):
    if False:
        return 10
    'Computes embeddings for each entry of sparse features sparse_features.\n\n  Args:\n    params: list of 2D tensors containing vector embeddings\n    sparse_features: 1D tensor of strings. Each entry is a string encoding of\n      dist_belief.SparseFeatures, and represents a variable length list of\n      feature ids, and optionally, corresponding weights values.\n    allow_weights: boolean to control whether the weights returned from the\n      SparseFeatures are used to multiply the embeddings.\n\n  Returns:\n    A tensor representing the combined embeddings for the sparse features.\n    For each entry s in sparse_features, the function looks up the embeddings\n    for each id and sums them into a single tensor weighing them by the\n    weight of each id. It returns a tensor with each entry of sparse_features\n    replaced by this combined embedding.\n  '
    if not isinstance(params, list):
        params = [params]
    sparse_features = tf.convert_to_tensor(sparse_features)
    (indices, ids, weights) = gen_parser_ops.unpack_syntax_net_sparse_features(sparse_features)
    embeddings = tf.nn.embedding_lookup(params, ids)
    if allow_weights:
        broadcast_weights_shape = tf.concat([tf.shape(weights), [1]], 0)
        embeddings *= tf.reshape(weights, broadcast_weights_shape)
    return tf.unsorted_segment_sum(embeddings, indices, tf.size(sparse_features))

class GreedyParser(object):
    """Builds a Chen & Manning style greedy neural net parser.

  Builds a graph with an optional reader op connected at one end and
  operations needed to train the network on the other. Supports multiple
  network instantiations sharing the same parameters and network topology.

  The following named nodes are added to the training and eval networks:
    epochs: a tensor containing the current epoch number
    cost: a tensor containing the current training step cost
    gold_actions: a tensor containing actions from gold decoding
    feature_endpoints: a list of sparse feature vectors
    logits: output of the final layer before computing softmax
  The training network also contains:
    train_op: an op that executes a single training step

  Typical usage:

  parser = graph_builder.GreedyParser(num_actions, num_features,
                                      num_feature_ids, embedding_sizes,
                                      hidden_layer_sizes)
  parser.AddTraining(task_context, batch_size=5)
  with tf.Session('local') as sess:
    # This works because the session uses the same default graph as the
    # GraphBuilder did.
    sess.run(parser.inits.values())
    while True:
      tf_epoch, _ = sess.run([parser.training['epoch'],
                              parser.training['train_op']])
      if tf_epoch[0] > 0:
        break
  """

    def __init__(self, num_actions, num_features, num_feature_ids, embedding_sizes, hidden_layer_sizes, seed=None, gate_gradients=False, use_locking=False, embedding_init=1.0, relu_init=0.0001, bias_init=0.2, softmax_init=0.0001, averaging_decay=0.9999, use_averaging=True, check_parameters=True, check_every=1, allow_feature_weights=False, only_train='', arg_prefix=None, **unused_kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the graph builder with parameters defining the network.\n\n    Args:\n      num_actions: int size of the set of parser actions\n      num_features: int list of dimensions of the feature vectors\n      num_feature_ids: int list of same length as num_features corresponding to\n        the sizes of the input feature spaces\n      embedding_sizes: int list of same length as num_features of the desired\n        embedding layer sizes\n      hidden_layer_sizes: int list of desired relu layer sizes; may be empty\n      seed: optional random initializer seed to enable reproducibility\n      gate_gradients: if True, gradient updates are computed synchronously,\n        ensuring consistency and reproducibility\n      use_locking: if True, use locking to avoid read-write contention when\n        updating Variables\n      embedding_init: sets the std dev of normal initializer of embeddings to\n        embedding_init / embedding_size ** .5\n      relu_init: sets the std dev of normal initializer of relu weights\n        to relu_init\n      bias_init: sets constant initializer of relu bias to bias_init\n      softmax_init: sets the std dev of normal initializer of softmax init\n        to softmax_init\n      averaging_decay: decay for exponential moving average when computing\n        averaged parameters, set to 1 to do vanilla averaging\n      use_averaging: whether to use moving averages of parameters during evals\n      check_parameters: whether to check for NaN/Inf parameters during\n        training\n      check_every: checks numerics every check_every steps.\n      allow_feature_weights: whether feature weights are allowed.\n      only_train: the comma separated set of parameter names to train. If empty,\n        all model parameters will be trained.\n      arg_prefix: prefix for context parameters.\n    '
        self._num_actions = num_actions
        self._num_features = num_features
        self._num_feature_ids = num_feature_ids
        self._embedding_sizes = embedding_sizes
        self._hidden_layer_sizes = hidden_layer_sizes
        self._seed = seed
        self._gate_gradients = gate_gradients
        self._use_locking = use_locking
        self._use_averaging = use_averaging
        self._check_parameters = check_parameters
        self._check_every = check_every
        self._allow_feature_weights = allow_feature_weights
        self._only_train = set(only_train.split(',')) if only_train else None
        self._feature_size = len(embedding_sizes)
        self._embedding_init = embedding_init
        self._relu_init = relu_init
        self._softmax_init = softmax_init
        self._arg_prefix = arg_prefix
        self.params = {}
        self.variables = {}
        self.inits = {}
        self.training = {}
        self.evaluation = {}
        self.saver = None
        self._averaging = {}
        self._averaging_decay = averaging_decay
        self._pretrained_embeddings = {}
        with tf.name_scope('params') as self._param_scope:
            self._relu_bias_init = tf.constant_initializer(bias_init)

    @property
    def embedding_size(self):
        if False:
            i = 10
            return i + 15
        size = 0
        for i in range(self._feature_size):
            size += self._num_features[i] * self._embedding_sizes[i]
        return size

    def _AddParam(self, shape, dtype, name, initializer=None, return_average=False):
        if False:
            while True:
                i = 10
        "Add a model parameter w.r.t. we expect to compute gradients.\n\n    _AddParam creates both regular parameters (usually for training) and\n    averaged nodes (usually for inference). It returns one or the other based\n    on the 'return_average' arg.\n\n    Args:\n      shape: int list, tensor shape of the parameter to create\n      dtype: tf.DataType, data type of the parameter\n      name: string, name of the parameter in the TF graph\n      initializer: optional initializer for the paramter\n      return_average: if False, return parameter otherwise return moving average\n\n    Returns:\n      parameter or averaged parameter\n    "
        if name not in self.params:
            step = tf.cast(self.GetStep(), tf.float32)
            with tf.name_scope(self._param_scope):
                self.params[name] = tf.get_variable(name, shape, dtype, initializer)
                param = self.params[name]
                if initializer is not None:
                    self.inits[name] = state_ops.init_variable(param, initializer)
                if self._averaging_decay == 1:
                    logging.info('Using vanilla averaging of parameters.')
                    ema = tf.train.ExponentialMovingAverage(decay=step / (step + 1.0), num_updates=None)
                else:
                    ema = tf.train.ExponentialMovingAverage(decay=self._averaging_decay, num_updates=step)
                self._averaging[name + '_avg_update'] = ema.apply([param])
                self.variables[name + '_avg_var'] = ema.average(param)
                self.inits[name + '_avg_init'] = state_ops.init_variable(ema.average(param), tf.zeros_initializer())
        return self.variables[name + '_avg_var'] if return_average else self.params[name]

    def GetStep(self):
        if False:
            for i in range(10):
                print('nop')

        def OnesInitializer(shape, dtype=tf.float32, partition_info=None):
            if False:
                while True:
                    i = 10
            return tf.ones(shape, dtype)
        return self._AddVariable([], tf.int32, 'step', OnesInitializer)

    def _AddVariable(self, shape, dtype, name, initializer=None):
        if False:
            while True:
                i = 10
        if name in self.variables:
            return self.variables[name]
        self.variables[name] = tf.get_variable(name, shape, dtype, initializer)
        if initializer is not None:
            self.inits[name] = state_ops.init_variable(self.variables[name], initializer)
        return self.variables[name]

    def _ReluWeightInitializer(self):
        if False:
            print('Hello World!')
        with tf.name_scope(self._param_scope):
            return tf.random_normal_initializer(stddev=self._relu_init, seed=self._seed)

    def _EmbeddingMatrixInitializer(self, index, embedding_size):
        if False:
            return 10
        if index in self._pretrained_embeddings:
            return self._pretrained_embeddings[index]
        else:
            return tf.random_normal_initializer(stddev=self._embedding_init / embedding_size ** 0.5, seed=self._seed)

    def _AddEmbedding(self, features, num_features, num_ids, embedding_size, index, return_average=False):
        if False:
            return 10
        'Adds an embedding matrix and passes the `features` vector through it.'
        embedding_matrix = self._AddParam([num_ids, embedding_size], tf.float32, 'embedding_matrix_%d' % index, self._EmbeddingMatrixInitializer(index, embedding_size), return_average=return_average)
        embedding = EmbeddingLookupFeatures(embedding_matrix, tf.reshape(features, [-1], name='feature_%d' % index), self._allow_feature_weights)
        return tf.reshape(embedding, [-1, num_features * embedding_size])

    def _BuildNetwork(self, feature_endpoints, return_average=False):
        if False:
            print('Hello World!')
        'Builds a feed-forward part of the net given features as input.\n\n    The network topology is already defined in the constructor, so multiple\n    calls to BuildForward build multiple networks whose parameters are all\n    shared. It is the source of the input features and the use of the output\n    that distinguishes each network.\n\n    Args:\n      feature_endpoints: tensors with input features to the network\n      return_average: whether to use moving averages as model parameters\n\n    Returns:\n      logits: output of the final layer before computing softmax\n    '
        assert len(feature_endpoints) == self._feature_size
        embeddings = []
        for i in range(self._feature_size):
            embeddings.append(self._AddEmbedding(feature_endpoints[i], self._num_features[i], self._num_feature_ids[i], self._embedding_sizes[i], i, return_average=return_average))
        last_layer = tf.concat(embeddings, 1)
        last_layer_size = self.embedding_size
        for (i, hidden_layer_size) in enumerate(self._hidden_layer_sizes):
            weights = self._AddParam([last_layer_size, hidden_layer_size], tf.float32, 'weights_%d' % i, self._ReluWeightInitializer(), return_average=return_average)
            bias = self._AddParam([hidden_layer_size], tf.float32, 'bias_%d' % i, self._relu_bias_init, return_average=return_average)
            last_layer = tf.nn.relu_layer(last_layer, weights, bias, name='layer_%d' % i)
            last_layer_size = hidden_layer_size
        softmax_weight = self._AddParam([last_layer_size, self._num_actions], tf.float32, 'softmax_weight', tf.random_normal_initializer(stddev=self._softmax_init, seed=self._seed), return_average=return_average)
        softmax_bias = self._AddParam([self._num_actions], tf.float32, 'softmax_bias', tf.zeros_initializer(), return_average=return_average)
        logits = tf.nn.xw_plus_b(last_layer, softmax_weight, softmax_bias, name='logits')
        return {'logits': logits}

    def _AddGoldReader(self, task_context, batch_size, corpus_name):
        if False:
            for i in range(10):
                print('nop')
        (features, epochs, gold_actions) = gen_parser_ops.gold_parse_reader(task_context, self._feature_size, batch_size, corpus_name=corpus_name, arg_prefix=self._arg_prefix)
        return {'gold_actions': tf.identity(gold_actions, name='gold_actions'), 'epochs': tf.identity(epochs, name='epochs'), 'feature_endpoints': features}

    def _AddDecodedReader(self, task_context, batch_size, transition_scores, corpus_name):
        if False:
            return 10
        (features, epochs, eval_metrics, documents) = gen_parser_ops.decoded_parse_reader(transition_scores, task_context, self._feature_size, batch_size, corpus_name=corpus_name, arg_prefix=self._arg_prefix)
        return {'eval_metrics': eval_metrics, 'epochs': tf.identity(epochs, name='epochs'), 'feature_endpoints': features, 'documents': documents}

    def _AddCostFunction(self, batch_size, gold_actions, logits):
        if False:
            return 10
        'Cross entropy plus L2 loss on weights and biases of the hidden layers.'
        dense_golden = BatchedSparseToDense(gold_actions, self._num_actions)
        cross_entropy = tf.div(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=dense_golden, logits=logits)), batch_size)
        regularized_params = [tf.nn.l2_loss(p) for (k, p) in self.params.items() if k.startswith('weights') or k.startswith('bias')]
        l2_loss = 0.0001 * tf.add_n(regularized_params) if regularized_params else 0
        return {'cost': tf.add(cross_entropy, l2_loss, name='cost')}

    def AddEvaluation(self, task_context, batch_size, evaluation_max_steps=300, corpus_name='documents'):
        if False:
            i = 10
            return i + 15
        'Builds the forward network only without the training operation.\n\n    Args:\n      task_context: file path from which to read the task context.\n      batch_size: batch size to request from reader op.\n      evaluation_max_steps: max number of parsing actions during evaluation,\n          only used in beam parsing.\n      corpus_name: name of the task input to read parses from.\n\n    Returns:\n      Dictionary of named eval nodes.\n    '

        def _AssignTransitionScores():
            if False:
                return 10
            return tf.assign(nodes['transition_scores'], nodes['logits'], validate_shape=False)

        def _Pass():
            if False:
                return 10
            return tf.constant(-1.0)
        unused_evaluation_max_steps = evaluation_max_steps
        with tf.name_scope('evaluation'):
            nodes = self.evaluation
            nodes['transition_scores'] = self._AddVariable([batch_size, self._num_actions], tf.float32, 'transition_scores', tf.constant_initializer(-1.0))
            nodes.update(self._AddDecodedReader(task_context, batch_size, nodes['transition_scores'], corpus_name))
            nodes.update(self._BuildNetwork(nodes['feature_endpoints'], return_average=self._use_averaging))
            nodes['eval_metrics'] = cf.with_dependencies([tf.cond(tf.greater(tf.size(nodes['logits']), 0), _AssignTransitionScores, _Pass)], nodes['eval_metrics'], name='eval_metrics')
        return nodes

    def _IncrementCounter(self, counter):
        if False:
            while True:
                i = 10
        return state_ops.assign_add(counter, 1, use_locking=True)

    def _AddLearningRate(self, initial_learning_rate, decay_steps):
        if False:
            i = 10
            return i + 15
        'Returns a learning rate that decays by 0.96 every decay_steps.\n\n    Args:\n      initial_learning_rate: initial value of the learning rate\n      decay_steps: decay by 0.96 every this many steps\n\n    Returns:\n      learning rate variable.\n    '
        step = self.GetStep()
        return cf.with_dependencies([self._IncrementCounter(step)], tf.train.exponential_decay(initial_learning_rate, step, decay_steps, 0.96, staircase=True))

    def AddPretrainedEmbeddings(self, index, embeddings_path, task_context):
        if False:
            i = 10
            return i + 15
        'Embeddings at the given index will be set to pretrained values.'

        def _Initializer(shape, dtype=tf.float32, partition_info=None):
            if False:
                return 10
            'Variable initializer that loads pretrained embeddings.'
            unused_dtype = dtype
            (seed1, seed2) = tf.get_seed(self._seed)
            t = gen_parser_ops.word_embedding_initializer(vectors=embeddings_path, task_context=task_context, embedding_init=self._embedding_init, cache_vectors_locally=False, seed=seed1, seed2=seed2)
            t.set_shape(shape)
            return t
        self._pretrained_embeddings[index] = _Initializer

    def AddTraining(self, task_context, batch_size, learning_rate=0.1, decay_steps=4000, momentum=0.9, corpus_name='documents'):
        if False:
            return 10
        'Builds a trainer to minimize the cross entropy cost function.\n\n    Args:\n      task_context: file path from which to read the task context\n      batch_size: batch size to request from reader op\n      learning_rate: initial value of the learning rate\n      decay_steps: decay learning rate by 0.96 every this many steps\n      momentum: momentum parameter used when training with momentum\n      corpus_name: name of the task input to read parses from\n\n    Returns:\n      Dictionary of named training nodes.\n    '
        with tf.name_scope('training'):
            nodes = self.training
            nodes.update(self._AddGoldReader(task_context, batch_size, corpus_name))
            nodes.update(self._BuildNetwork(nodes['feature_endpoints'], return_average=False))
            nodes.update(self._AddCostFunction(batch_size, nodes['gold_actions'], nodes['logits']))
            if self._only_train:
                trainable_params = [v for (k, v) in self.params.iteritems() if k in self._only_train]
            else:
                trainable_params = self.params.values()
            lr = self._AddLearningRate(learning_rate, decay_steps)
            optimizer = tf.train.MomentumOptimizer(lr, momentum, use_locking=self._use_locking)
            train_op = optimizer.minimize(nodes['cost'], var_list=trainable_params)
            for param in trainable_params:
                slot = optimizer.get_slot(param, 'momentum')
                self.inits[slot.name] = state_ops.init_variable(slot, tf.zeros_initializer())
                self.variables[slot.name] = slot
            numerical_checks = [tf.check_numerics(param, message='Parameter is not finite.') for param in trainable_params if param.dtype.base_dtype in [tf.float32, tf.float64]]
            check_op = tf.group(*numerical_checks)
            avg_update_op = tf.group(*self._averaging.values())
            train_ops = [train_op]
            if self._check_parameters:
                train_ops.append(check_op)
            if self._use_averaging:
                train_ops.append(avg_update_op)
            nodes['train_op'] = tf.group(*train_ops, name='train_op')
        return nodes

    def AddSaver(self, slim_model=False):
        if False:
            print('Hello World!')
        'Adds ops to save and restore model parameters.\n\n    Args:\n      slim_model: whether only averaged variables are saved.\n\n    Returns:\n      the saver object.\n    '
        with tf.name_scope(None):
            variables_to_save = self.params.copy()
            variables_to_save.update(self.variables)
            if slim_model:
                for key in variables_to_save.keys():
                    if not key.endswith('avg_var'):
                        del variables_to_save[key]
            self.saver = tf.train.Saver(variables_to_save, builder=tf_saver.BaseSaverBuilder())
        return self.saver