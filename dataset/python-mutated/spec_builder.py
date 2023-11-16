"""Utils for building DRAGNN specs."""
import tensorflow as tf
from dragnn.protos import spec_pb2
from dragnn.python import lexicon
from syntaxnet.ops import gen_parser_ops
from syntaxnet.util import check

class ComponentSpecBuilder(object):
    """Wrapper to help construct SyntaxNetComponent specifications.

  This class will help make sure that ComponentSpec's are consistent with the
  expectations of the SyntaxNet Component backend. It contains defaults used to
  create LinkFeatureChannel specifications according to the network_unit and
  transition_system of the source compnent.  It also encapsulates common recipes
  for hooking up FML and translators.

  Attributes:
    spec: The dragnn.ComponentSpec proto.
  """

    def __init__(self, name, builder='DynamicComponentBuilder', backend='SyntaxNetComponent'):
        if False:
            while True:
                i = 10
        'Initializes the ComponentSpec with some defaults for SyntaxNet.\n\n    Args:\n      name: The name of this Component in the pipeline.\n      builder: The component builder type.\n      backend: The component backend type.\n    '
        self.spec = spec_pb2.ComponentSpec(name=name, backend=self.make_module(backend), component_builder=self.make_module(builder))

    def make_module(self, name, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Forwards kwargs to easily created a RegisteredModuleSpec.\n\n    Note: all kwargs should be string-valued.\n\n    Args:\n      name: The registered name of the module.\n      **kwargs: Proto fields to be specified in the module.\n\n    Returns:\n      Newly created RegisteredModuleSpec.\n    '
        return spec_pb2.RegisteredModuleSpec(registered_name=name, parameters=kwargs)

    def default_source_layer(self):
        if False:
            return 10
        'Returns the default source_layer setting for this ComponentSpec.\n\n    Usually links are intended for a specific layer in the network unit.\n    For common network units, this returns the hidden layer intended\n    to be read by recurrent and cross-component connections.\n\n    Returns:\n      String name of default network layer.\n\n    Raises:\n      ValueError: if no default is known for the given setup.\n    '
        for (network, default_layer) in [('FeedForwardNetwork', 'layer_0'), ('LayerNormBasicLSTMNetwork', 'state_h_0'), ('LSTMNetwork', 'layer_0'), ('IdentityNetwork', 'input_embeddings')]:
            if self.spec.network_unit.registered_name.endswith(network):
                return default_layer
        raise ValueError('No default source for network unit: %s' % self.spec.network_unit)

    def default_token_translator(self):
        if False:
            print('Hello World!')
        'Returns the default source_translator setting for token representations.\n\n    Most links are token-based: given a target token index, retrieve a learned\n    representation for that token from this component. This depends on the\n    transition system; e.g. we should make sure that left-to-right sequence\n    models reverse the incoming token index when looking up representations from\n    a right-to-left model.\n\n    Returns:\n      String name of default translator for this transition system.\n\n    Raises:\n      ValueError: if no default is known for the given setup.\n    '
        transition_spec = self.spec.transition_system
        if transition_spec.registered_name == 'arc-standard':
            return 'shift-reduce-step'
        if transition_spec.registered_name in ('shift-only', 'tagger', 'morpher', 'lm-transitions', 'dependency-label', 'category'):
            if 'left_to_right' in transition_spec.parameters:
                if transition_spec.parameters['left_to_right'] == 'false':
                    return 'reverse-token'
            return 'identity'
        raise ValueError('Invalid transition spec: %s' % str(transition_spec))

    def add_token_link(self, source=None, source_layer=None, **kwargs):
        if False:
            return 10
        "Adds a link to source's token representations using default settings.\n\n    Constructs a LinkedFeatureChannel proto and adds it to the spec, using\n    defaults to assign the name, component, translator, and layer of the\n    channel.  The user must provide fml and embedding_dim.\n\n    Args:\n      source: SyntaxComponentBuilder object to pull representations from.\n      source_layer: Optional override for a source layer instead of the default.\n      **kwargs: Forwarded arguments to the LinkedFeatureChannel proto.\n    "
        if source_layer is None:
            source_layer = source.default_source_layer()
        self.spec.linked_feature.add(name=source.spec.name, source_component=source.spec.name, source_layer=source_layer, source_translator=source.default_token_translator(), **kwargs)

    def add_rnn_link(self, source_layer=None, **kwargs):
        if False:
            print('Hello World!')
        'Adds a recurrent link to this component using default settings.\n\n    This adds the connection to the previous time step only to the network.  It\n    constructs a LinkedFeatureChannel proto and adds it to the spec, using\n    defaults to assign the name, component, translator, and layer of the\n    channel.  The user must provide the embedding_dim only.\n\n    Args:\n      source_layer: Optional override for a source layer instead of the default.\n      **kwargs: Forwarded arguments to the LinkedFeatureChannel proto.\n    '
        if source_layer is None:
            source_layer = self.default_source_layer()
        self.spec.linked_feature.add(name='rnn', source_layer=source_layer, source_component=self.spec.name, source_translator='history', fml='constant', **kwargs)

    def set_transition_system(self, *args, **kwargs):
        if False:
            return 10
        'Shorthand to set transition_system using kwargs.'
        self.spec.transition_system.CopyFrom(self.make_module(*args, **kwargs))

    def set_network_unit(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Shorthand to set network_unit using kwargs.'
        self.spec.network_unit.CopyFrom(self.make_module(*args, **kwargs))

    def add_fixed_feature(self, **kwargs):
        if False:
            while True:
                i = 10
        'Shorthand to add a fixed_feature using kwargs.'
        self.spec.fixed_feature.add(**kwargs)

    def add_link(self, source, source_layer=None, source_translator='identity', name=None, **kwargs):
        if False:
            while True:
                i = 10
        'Add a link using default naming and layers only.'
        if source_layer is None:
            source_layer = source.default_source_layer()
        if name is None:
            name = source.spec.name
        self.spec.linked_feature.add(source_component=source.spec.name, source_layer=source_layer, name=name, source_translator=source_translator, **kwargs)

    def fill_from_resources(self, resource_path, tf_master=''):
        if False:
            i = 10
            return i + 15
        "Fills in feature sizes and vocabularies using SyntaxNet lexicon.\n\n    Must be called before the spec is ready to be used to build TensorFlow\n    graphs. Requires a SyntaxNet lexicon built at the resource_path. Using the\n    lexicon, this will call the SyntaxNet custom ops to return the number of\n    features and vocabulary sizes based on the FML specifications and the\n    lexicons. It will also compute the number of actions of the transition\n    system.\n\n    This will often CHECK-fail if the spec doesn't correspond to a valid\n    transition system or feature setup.\n\n    Args:\n      resource_path: Path to the lexicon.\n      tf_master: TensorFlow master executor (string, defaults to '' to use the\n        local instance).\n    "
        check.IsTrue(self.spec.transition_system.registered_name, 'Set a transition system before calling fill_from_resources().')
        context = lexicon.create_lexicon_context(resource_path)
        for resource in self.spec.resource:
            context.input.add(name=resource.name).part.add(file_pattern=resource.part[0].file_pattern)
        for (key, value) in self.spec.transition_system.parameters.iteritems():
            context.parameter.add(name=key, value=value)
        context.parameter.add(name='brain_parser_embedding_dims', value=';'.join([str(x.embedding_dim) for x in self.spec.fixed_feature]))
        context.parameter.add(name='brain_parser_features', value=';'.join([x.fml for x in self.spec.fixed_feature]))
        context.parameter.add(name='brain_parser_predicate_maps', value=';'.join(['' for x in self.spec.fixed_feature]))
        context.parameter.add(name='brain_parser_embedding_names', value=';'.join([x.name for x in self.spec.fixed_feature]))
        context.parameter.add(name='brain_parser_transition_system', value=self.spec.transition_system.registered_name)
        with tf.Session(tf_master) as sess:
            (feature_sizes, domain_sizes, _, num_actions) = sess.run(gen_parser_ops.feature_size(task_context_str=str(context)))
            self.spec.num_actions = int(num_actions)
            for i in xrange(len(feature_sizes)):
                self.spec.fixed_feature[i].size = int(feature_sizes[i])
                self.spec.fixed_feature[i].vocabulary_size = int(domain_sizes[i])
        for i in xrange(len(self.spec.linked_feature)):
            self.spec.linked_feature[i].size = len(self.spec.linked_feature[i].fml.split(' '))
        del self.spec.resource[:]
        for resource in context.input:
            self.spec.resource.add(name=resource.name).part.add(file_pattern=resource.part[0].file_pattern)

def complete_master_spec(master_spec, lexicon_corpus, output_path, tf_master=''):
    if False:
        print('Hello World!')
    "Finishes a MasterSpec that defines the network config.\n\n  Given a MasterSpec that defines the DRAGNN architecture, completes the spec so\n  that it can be used to build a DRAGNN graph and run training/inference.\n\n  Args:\n    master_spec: MasterSpec.\n    lexicon_corpus: the corpus to be used with the LexiconBuilder.\n    output_path: directory to save resources to.\n    tf_master: TensorFlow master executor (string, defaults to '' to use the\n      local instance).\n\n  Returns:\n    None, since the spec is changed in-place.\n  "
    if lexicon_corpus:
        lexicon.build_lexicon(output_path, lexicon_corpus)
    for (i, spec) in enumerate(master_spec.component):
        builder = ComponentSpecBuilder(spec.name)
        builder.spec = spec
        builder.fill_from_resources(output_path, tf_master=tf_master)
        master_spec.component[i].CopyFrom(builder.spec)

def default_targets_from_spec(spec):
    if False:
        i = 10
        return i + 15
    "Constructs a default set of TrainTarget protos from a DRAGNN spec.\n\n  For each component in the DRAGNN spec, it adds a training target for that\n  component's oracle. It also stops unrolling the graph with that component.  It\n  skips any 'shift-only' transition systems which have no oracle. E.g.: if there\n  are three components, a 'shift-only', a 'tagger', and a 'arc-standard', it\n  will construct two training targets, one for the tagger and one for the\n  arc-standard parser.\n\n  Arguments:\n    spec: DRAGNN spec.\n\n  Returns:\n    List of TrainTarget protos.\n  "
    component_targets = [spec_pb2.TrainTarget(name=component.name, max_index=idx + 1, unroll_using_oracle=[False] * idx + [True]) for (idx, component) in enumerate(spec.component) if not component.transition_system.registered_name.endswith('shift-only')]
    return component_targets