"""Component builders for non-recurrent networks in DRAGNN."""
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from dragnn.python import component
from dragnn.python import dragnn_ops
from dragnn.python import network_units
from syntaxnet.util import check

def fetch_linked_embedding(comp, network_states, feature_spec):
    if False:
        i = 10
        return i + 15
    "Looks up linked embeddings in other components.\n\n  Args:\n    comp: ComponentBuilder object with respect to which the feature is to be\n        fetched\n    network_states: dictionary of NetworkState objects\n    feature_spec: FeatureSpec proto for the linked feature to be looked up\n\n  Returns:\n    NamedTensor containing the linked feature tensor\n\n  Raises:\n    NotImplementedError: if a linked feature with source translator other than\n        'identity' is configured.\n    RuntimeError: if a recurrent linked feature is configured.\n  "
    if feature_spec.source_translator != 'identity':
        raise NotImplementedError(feature_spec.source_translator)
    if feature_spec.source_component == comp.name:
        raise RuntimeError('Recurrent linked features are not supported in bulk extraction.')
    tf.logging.info('[%s] Adding linked feature "%s"', comp.name, feature_spec.name)
    source = comp.master.lookup_component[feature_spec.source_component]
    return network_units.NamedTensor(network_states[source.name].activations[feature_spec.source_layer].bulk_tensor, feature_spec.name)

def _validate_embedded_fixed_features(comp):
    if False:
        return 10
    'Checks that the embedded fixed features of |comp| are set up properly.'
    for feature in comp.spec.fixed_feature:
        check.Gt(feature.embedding_dim, 0, 'Embeddings requested for non-embedded feature: %s' % feature)
        if feature.is_constant:
            check.IsTrue(feature.HasField('pretrained_embedding_matrix'), 'Constant embeddings must be pretrained: %s' % feature)

def fetch_differentiable_fixed_embeddings(comp, state, stride, during_training):
    if False:
        for i in range(10):
            print('nop')
    'Looks up fixed features with separate, differentiable, embedding lookup.\n\n  Args:\n    comp: Component whose fixed features we wish to look up.\n    state: live MasterState object for the component.\n    stride: Tensor containing current batch * beam size.\n    during_training: True if this is being called from a training code path.\n      This controls, e.g., the use of feature ID dropout.\n\n  Returns:\n    state handle: updated state handle to be used after this call\n    fixed_embeddings: list of NamedTensor objects\n  '
    _validate_embedded_fixed_features(comp)
    num_channels = len(comp.spec.fixed_feature)
    if not num_channels:
        return (state.handle, [])
    (state.handle, indices, ids, weights, num_steps) = dragnn_ops.bulk_fixed_features(state.handle, component=comp.name, num_channels=num_channels)
    fixed_embeddings = []
    for (channel, feature_spec) in enumerate(comp.spec.fixed_feature):
        differentiable_or_constant = 'constant' if feature_spec.is_constant else 'differentiable'
        tf.logging.info('[%s] Adding %s fixed feature "%s"', comp.name, differentiable_or_constant, feature_spec.name)
        if during_training and feature_spec.dropout_id >= 0:
            (ids[channel], weights[channel]) = network_units.apply_feature_id_dropout(ids[channel], weights[channel], feature_spec)
        size = stride * num_steps * feature_spec.size
        fixed_embedding = network_units.embedding_lookup(comp.get_variable(network_units.fixed_embeddings_name(channel)), indices[channel], ids[channel], weights[channel], size)
        if feature_spec.is_constant:
            fixed_embedding = tf.stop_gradient(fixed_embedding)
        fixed_embeddings.append(network_units.NamedTensor(fixed_embedding, feature_spec.name))
    return (state.handle, fixed_embeddings)

def fetch_fast_fixed_embeddings(comp, state, pad_to_batch=None, pad_to_steps=None):
    if False:
        while True:
            i = 10
    'Looks up fixed features with fast, non-differentiable, op.\n\n  Since BulkFixedEmbeddings is non-differentiable with respect to the\n  embeddings, the idea is to call this function only when the graph is\n  not being used for training. If the function is being called with fixed step\n  and batch sizes, it will use the most efficient possible extractor.\n\n  Args:\n    comp: Component whose fixed features we wish to look up.\n    state: live MasterState object for the component.\n    pad_to_batch: Optional; the number of batch elements to pad to.\n    pad_to_steps: Optional; the number of steps to pad to.\n\n  Returns:\n    state handle: updated state handle to be used after this call\n    fixed_embeddings: list of NamedTensor objects\n  '
    _validate_embedded_fixed_features(comp)
    num_channels = len(comp.spec.fixed_feature)
    if not num_channels:
        return (state.handle, [])
    tf.logging.info('[%s] Adding %d fast fixed features', comp.name, num_channels)
    features = [comp.get_variable(network_units.fixed_embeddings_name(c)) for c in range(num_channels)]
    if pad_to_batch is not None and pad_to_steps is not None:
        (state.handle, bulk_embeddings, _) = dragnn_ops.bulk_embed_fixed_features(state.handle, features, component=comp.name, pad_to_batch=pad_to_batch, pad_to_steps=pad_to_steps)
    else:
        (state.handle, bulk_embeddings, _) = dragnn_ops.bulk_fixed_embeddings(state.handle, features, component=comp.name)
    bulk_embeddings = network_units.NamedTensor(bulk_embeddings, 'bulk-%s-fixed-features' % comp.name)
    return (state.handle, [bulk_embeddings])

def fetch_dense_ragged_embeddings(comp, state):
    if False:
        while True:
            i = 10
    'Gets embeddings in RaggedTensor format.'
    _validate_embedded_fixed_features(comp)
    num_channels = len(comp.spec.fixed_feature)
    if not num_channels:
        return (state.handle, [])
    tf.logging.info('[%s] Adding %d fast fixed features', comp.name, num_channels)
    features = [comp.get_variable(network_units.fixed_embeddings_name(c)) for c in range(num_channels)]
    (state.handle, data, offsets) = dragnn_ops.bulk_embed_dense_fixed_features(state.handle, features, component=comp.name)
    data = network_units.NamedTensor(data, 'dense-%s-data' % comp.name)
    offsets = network_units.NamedTensor(offsets, 'dense-%s-offsets' % comp.name)
    return (state.handle, [data, offsets])

def extract_fixed_feature_ids(comp, state, stride):
    if False:
        i = 10
        return i + 15
    'Extracts fixed feature IDs.\n\n  Args:\n    comp: Component whose fixed feature IDs we wish to extract.\n    state: Live MasterState object for the component.\n    stride: Tensor containing current batch * beam size.\n\n  Returns:\n    state handle: Updated state handle to be used after this call.\n    ids: List of [stride * num_steps, 1] feature IDs per channel.  Missing IDs\n         (e.g., due to batch padding) are set to -1.\n  '
    num_channels = len(comp.spec.fixed_feature)
    if not num_channels:
        return (state.handle, [])
    for feature_spec in comp.spec.fixed_feature:
        check.Eq(feature_spec.size, 1, 'All features must have size=1')
        check.Lt(feature_spec.embedding_dim, 0, 'All features must be non-embedded')
    (state.handle, indices, ids, _, num_steps) = dragnn_ops.bulk_fixed_features(state.handle, component=comp.name, num_channels=num_channels)
    size = stride * num_steps
    fixed_ids = []
    for (channel, feature_spec) in enumerate(comp.spec.fixed_feature):
        tf.logging.info('[%s] Adding fixed feature IDs "%s"', comp.name, feature_spec.name)
        sums = tf.unsorted_segment_sum(ids[channel] + 1, indices[channel], size) - 1
        sums = tf.expand_dims(sums, axis=1)
        fixed_ids.append(network_units.NamedTensor(sums, feature_spec.name, dim=1))
    return (state.handle, fixed_ids)

def update_network_states(comp, tensors, network_states, stride):
    if False:
        return 10
    'Stores Tensor objects corresponding to layer outputs.\n\n  For use in subsequent tasks.\n\n  Args:\n    comp: Component for which the tensor handles are being stored.\n    tensors: list of Tensors to store\n    network_states: dictionary of component NetworkState objects\n    stride: stride of the stored tensor.\n  '
    network_state = network_states[comp.name]
    with tf.name_scope(comp.name + '/stored_act'):
        for (index, network_tensor) in enumerate(tensors):
            network_state.activations[comp.network.layers[index].name] = network_units.StoredActivations(tensor=network_tensor, stride=stride, dim=comp.network.layers[index].dim)

def build_cross_entropy_loss(logits, gold):
    if False:
        for i in range(10):
            print('nop')
    'Constructs a cross entropy from logits and one-hot encoded gold labels.\n\n  Supports skipping rows where the gold label is the magic -1 value.\n\n  Args:\n    logits: float Tensor of scores.\n    gold: int Tensor of gold label ids.\n\n  Returns:\n    cost, correct, total: the total cost, the total number of correctly\n        predicted labels, and the total number of valid labels.\n  '
    valid = tf.reshape(tf.where(tf.greater(gold, -1)), [-1])
    gold = tf.gather(gold, valid)
    logits = tf.gather(logits, valid)
    correct = tf.reduce_sum(tf.to_int32(tf.nn.in_top_k(logits, gold, 1)))
    total = tf.size(gold)
    with tf.control_dependencies([tf.assert_positive(total)]):
        cost = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(gold, tf.int64), logits=logits)) / tf.cast(total, tf.float32)
    return (cost, correct, total)

class BulkFeatureExtractorComponentBuilder(component.ComponentBuilderBase):
    """A component builder to bulk extract features.

  Both fixed and linked features are supported, with some restrictions:
  1. Fixed features may not be recurrent. Fixed features are extracted along the
     gold path, which does not work during inference.
  2. Linked features may not be recurrent and are 'untranslated'. For now,
     linked features are extracted without passing them through any transition
     system or source translator.
  """

    def build_greedy_training(self, state, network_states):
        if False:
            while True:
                i = 10
        "Extracts features and advances a batch using the oracle path.\n\n    Args:\n      state: MasterState from the 'AdvanceMaster' op that advances the\n          underlying master to this component.\n      network_states: dictionary of component NetworkState objects\n\n    Returns:\n      state handle: final state after advancing\n      cost: regularization cost, possibly associated with embedding matrices\n      correct: since no gold path is available, 0.\n      total: since no gold path is available, 0.\n    "
        logging.info('Building component: %s', self.spec.name)
        stride = state.current_batch_size * self.training_beam_size
        self.network.pre_create(stride)
        with tf.variable_scope(self.name, reuse=True):
            (state.handle, fixed_embeddings) = fetch_differentiable_fixed_embeddings(self, state, stride, True)
        linked_embeddings = [fetch_linked_embedding(self, network_states, spec) for spec in self.spec.linked_feature]
        with tf.variable_scope(self.name, reuse=True):
            tensors = self.network.create(fixed_embeddings, linked_embeddings, None, None, True, stride=stride)
        update_network_states(self, tensors, network_states, stride)
        cost = self.add_regularizer(tf.constant(0.0))
        (correct, total) = (tf.constant(0), tf.constant(0))
        return (state.handle, cost, correct, total)

    def build_post_restore_hook(self):
        if False:
            print('Hello World!')
        'Builds a graph that should be executed after the restore op.\n\n    This graph is intended to be run once, before the inference pipeline is\n    run.\n\n    Returns:\n      setup_op - An op that, when run, guarantees all setup ops will run.\n    '
        logging.info('Building restore hook for component: %s', self.spec.name)
        with tf.variable_scope(self.name):
            if callable(getattr(self.network, 'build_post_restore_hook', None)):
                return [self.network.build_post_restore_hook()]
            else:
                return []

    def build_greedy_inference(self, state, network_states, during_training=False):
        if False:
            print('Hello World!')
        "Extracts features and advances a batch using the oracle path.\n\n    NOTE(danielandor) For now this method cannot be called during training.\n    That is to say, unroll_using_oracle for this component must be set to true.\n    This will be fixed by separating train_with_oracle and train_with_inference.\n\n    Args:\n      state: MasterState from the 'AdvanceMaster' op that advances the\n          underlying master to this component.\n      network_states: dictionary of component NetworkState objects\n      during_training: whether the graph is being constructed during training\n\n    Returns:\n      state handle: final state after advancing\n    "
        logging.info('Building component: %s', self.spec.name)
        if during_training:
            stride = state.current_batch_size * self.training_beam_size
        else:
            stride = state.current_batch_size * self.inference_beam_size
        self.network.pre_create(stride)
        with tf.variable_scope(self.name, reuse=True):
            if during_training:
                (state.handle, fixed_embeddings) = fetch_differentiable_fixed_embeddings(self, state, stride, during_training)
            elif 'use_densors' in self.spec.network_unit.parameters:
                (state.handle, fixed_embeddings) = fetch_dense_ragged_embeddings(self, state)
            elif 'padded_batch_size' in self.spec.network_unit.parameters and 'padded_sentence_length' in self.spec.network_unit.parameters:
                (state.handle, fixed_embeddings) = fetch_fast_fixed_embeddings(self, state, pad_to_batch=-1, pad_to_steps=int(self.spec.network_unit.parameters['padded_sentence_length']))
            else:
                (state.handle, fixed_embeddings) = fetch_fast_fixed_embeddings(self, state)
        linked_embeddings = [fetch_linked_embedding(self, network_states, spec) for spec in self.spec.linked_feature]
        with tf.variable_scope(self.name, reuse=True):
            tensors = self.network.create(fixed_embeddings, linked_embeddings, None, None, during_training=during_training, stride=stride)
        update_network_states(self, tensors, network_states, stride)
        self._add_runtime_hooks()
        return state.handle

class BulkFeatureIdExtractorComponentBuilder(component.ComponentBuilderBase):
    """A component builder to bulk extract feature IDs.

  This is a variant of BulkFeatureExtractorComponentBuilder that only supports
  fixed features, and extracts raw feature IDs instead of feature embeddings.
  Since the extracted feature IDs are integers, the results produced by this
  component are in general not differentiable.
  """

    def __init__(self, master, component_spec):
        if False:
            while True:
                i = 10
        'Initializes the feature ID extractor component.\n\n    Args:\n      master: dragnn.MasterBuilder object.\n      component_spec: dragnn.ComponentSpec proto to be built.\n    '
        super(BulkFeatureIdExtractorComponentBuilder, self).__init__(master, component_spec)
        check.Eq(len(self.spec.linked_feature), 0, 'Linked features are forbidden')
        for feature_spec in self.spec.fixed_feature:
            check.Lt(feature_spec.embedding_dim, 0, 'Features must be non-embedded: %s' % feature_spec)

    def build_greedy_training(self, state, network_states):
        if False:
            for i in range(10):
                print('nop')
        'See base class.'
        state.handle = self._extract_feature_ids(state, network_states, True)
        cost = self.add_regularizer(tf.constant(0.0))
        (correct, total) = (tf.constant(0), tf.constant(0))
        return (state.handle, cost, correct, total)

    def build_greedy_inference(self, state, network_states, during_training=False):
        if False:
            i = 10
            return i + 15
        'See base class.'
        handle = self._extract_feature_ids(state, network_states, during_training)
        self._add_runtime_hooks()
        return handle

    def _extract_feature_ids(self, state, network_states, during_training):
        if False:
            i = 10
            return i + 15
        "Extracts feature IDs and advances a batch using the oracle path.\n\n    Args:\n      state: MasterState from the 'AdvanceMaster' op that advances the\n          underlying master to this component.\n      network_states: Dictionary of component NetworkState objects.\n      during_training: Whether the graph is being constructed during training.\n\n    Returns:\n      state handle: Final state after advancing.\n    "
        logging.info('Building component: %s', self.spec.name)
        if during_training:
            stride = state.current_batch_size * self.training_beam_size
        else:
            stride = state.current_batch_size * self.inference_beam_size
        self.network.pre_create(stride)
        with tf.variable_scope(self.name, reuse=True):
            (state.handle, ids) = extract_fixed_feature_ids(self, state, stride)
        with tf.variable_scope(self.name, reuse=True):
            tensors = self.network.create(ids, [], None, None, during_training, stride=stride)
        update_network_states(self, tensors, network_states, stride)
        return state.handle

class BulkAnnotatorComponentBuilder(component.ComponentBuilderBase):
    """A component builder to bulk annotate or compute the cost of a gold path.

  This component can be used with features that don't depend on the
  transition system state.

  Since no feature extraction is performed, only non-recurrent
  'identity' linked features are supported.

  If a FeedForwardNetwork is configured with no hidden units, this component
  acts as a 'bulk softmax' component.
  """

    def build_greedy_training(self, state, network_states):
        if False:
            return 10
        "Advances a batch using oracle paths, returning the overall CE cost.\n\n    Args:\n      state: MasterState from the 'AdvanceMaster' op that advances the\n          underlying master to this component.\n      network_states: dictionary of component NetworkState objects\n\n    Returns:\n      (state handle, cost, correct, total): TF ops corresponding to the final\n          state after unrolling, the total cost, the total number of correctly\n          predicted actions, and the total number of actions.\n\n    Raises:\n      RuntimeError: if fixed features are configured.\n    "
        logging.info('Building component: %s', self.spec.name)
        if self.spec.fixed_feature:
            raise RuntimeError('Fixed features are not compatible with bulk annotation. Use the "bulk-features" component instead.')
        linked_embeddings = [fetch_linked_embedding(self, network_states, spec) for spec in self.spec.linked_feature]
        stride = state.current_batch_size * self.training_beam_size
        self.network.pre_create(stride)
        with tf.variable_scope(self.name, reuse=True):
            network_tensors = self.network.create([], linked_embeddings, None, None, True, stride)
        update_network_states(self, network_tensors, network_states, stride)
        (state.handle, gold) = dragnn_ops.bulk_advance_from_oracle(state.handle, component=self.name)
        (cost, correct, total) = self.network.compute_bulk_loss(stride, network_tensors, gold)
        if cost is None:
            logits = self.network.get_logits(network_tensors)
            (cost, correct, total) = build_cross_entropy_loss(logits, gold)
        cost = self.add_regularizer(cost)
        return (state.handle, cost, correct, total)

    def build_greedy_inference(self, state, network_states, during_training=False):
        if False:
            for i in range(10):
                print('nop')
        "Annotates a batch of documents using network scores.\n\n    Args:\n      state: MasterState from the 'AdvanceMaster' op that advances the\n          underlying master to this component.\n      network_states: dictionary of component NetworkState objects\n      during_training: whether the graph is being constructed during training\n\n    Returns:\n      Handle to the state once inference is complete for this Component.\n\n    Raises:\n      RuntimeError: if fixed features are configured\n    "
        logging.info('Building component: %s', self.spec.name)
        if self.spec.fixed_feature:
            raise RuntimeError('Fixed features are not compatible with bulk annotation. Use the "bulk-features" component instead.')
        linked_embeddings = [fetch_linked_embedding(self, network_states, spec) for spec in self.spec.linked_feature]
        if during_training:
            stride = state.current_batch_size * self.training_beam_size
        else:
            stride = state.current_batch_size * self.inference_beam_size
        self.network.pre_create(stride)
        with tf.variable_scope(self.name, reuse=True):
            network_tensors = self.network.create([], linked_embeddings, None, None, during_training, stride)
        update_network_states(self, network_tensors, network_states, stride)
        logits = self.network.get_bulk_predictions(stride, network_tensors)
        if logits is None:
            logits = self.network.get_logits(network_tensors)
            logits = tf.cond(self.locally_normalize, lambda : tf.nn.log_softmax(logits), lambda : logits)
            if self._output_as_probabilities:
                logits = tf.nn.softmax(logits)
        handle = dragnn_ops.bulk_advance_from_prediction(state.handle, logits, component=self.name)
        self._add_runtime_hooks()
        return handle