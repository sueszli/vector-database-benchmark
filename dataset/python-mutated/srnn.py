"""SRNN classes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple
import functools
import sonnet as snt
import tensorflow as tf
from fivo.models import base
SRNNState = namedtuple('SRNNState', 'rnn_state latent_encoded')

class SRNN(object):
    """Implementation of a Stochastic Recurrent Neural Network (SRNN).

  Introduced in "Sequential Neural Models with Stochastic Layers"
  by Fraccaro et al. https://arxiv.org/pdf/1605.07571.pdf.

  The SRNN is a sequence model similar to an RNN that uses stochastic latent
  variables to improve its representational power. It can be thought of as a
  sequential analogue to the variational auto-encoder (VAE).

  The SRNN has a deterministic RNN as its backbone, represented by the
  sequence of RNN hidden states h_t. The latent state is conditioned on
  the deterministic RNN states and previous latent state. Unlike the VRNN, the
  the RNN state is not conditioned on the previous latent state. The latent
  states have a Markov structure and it is assumed that
  p(z_t | z_{1:t-1}) = p(z_t | z_{t-1}).

  In this implementation of the SRNN the latent state z_t is Gaussian. The
  model's prior over z_t (also called the transition distribution) is
  distributed as Normal(mu_t, diag(sigma_t^2)) where mu_t and sigma_t are the
  mean and standard deviation output from a fully connected network that accepts
  the rnn hidden state h_t and previous latent state z_{t-1} as input.

  The emission distribution p(x_t|z_t, h_t) is conditioned on the latent state
  z_t as well as the current RNN hidden state h_t via a fully connected network.

  To increase the modeling power of the SRNN, two additional networks are
  used to extract features from the data and the latent state. Those networks
  are called data_encoder and latent_encoder respectively.

  For an example of how to call the SRNN's methods see sample_step.

  There are a few differences between this exposition and the paper. The main
  goal was to be consistent with the VRNN code. A few components are renamed.
  The backward RNN for approximating the posterior, g_phi_a in the paper, is the
  rev_rnn_cell. The forward RNN that conditions the latent distribution, d in
  the paper, is the rnn_cell. The paper doesn't name the NN's that serve as
  feature extractors, and we name them here as the data_encoder and
  latent_encoder.
  """

    def __init__(self, rnn_cell, data_encoder, latent_encoder, transition, emission, random_seed=None):
        if False:
            for i in range(10):
                print('nop')
        "Create a SRNN.\n\n    Args:\n      rnn_cell: A subclass of tf.nn.rnn_cell.RNNCell that will form the\n        deterministic backbone of the SRNN. The inputs to the RNN will be the\n        the encoded input of the current timestep, a Tensor of shape\n        [batch_size, encoded_data_size].\n      data_encoder: A callable that accepts a batch of data x_t and\n        'encodes' it, e.g. runs it through a fully connected network. Must\n        accept as argument the inputs x_t, a Tensor of the shape\n        [batch_size, data_size] and return a Tensor of shape\n        [batch_size, encoded_data_size]. This callable will be called multiple\n        times in the SRNN cell so if scoping is not handled correctly then\n        multiple copies of the variables in this network could be made. It is\n        recommended to use a snt.nets.MLP module, which takes care of this for\n        you.\n      latent_encoder: A callable that accepts a latent state z_t and\n        'encodes' it, e.g. runs it through a fully connected network. Must\n        accept as argument a Tensor of shape [batch_size, latent_size] and\n        return a Tensor of shape [batch_size, encoded_latent_size].\n        This callable must also have the property 'output_size' defined,\n        returning encoded_latent_size.\n      transition: A callable that implements the transition distribution\n        p(z_t|h_t, z_t-1). Must accept as argument the previous RNN hidden state\n        and previous encoded latent state then return a tf.distributions.Normal\n        distribution conditioned on the input.\n      emission: A callable that implements the emission distribution\n        p(x_t|z_t, h_t). Must accept as arguments the encoded latent state\n        and the RNN hidden state and return a subclass of\n        tf.distributions.Distribution that can be used to evaluate the logprob\n        of the targets.\n      random_seed: The seed for the random ops. Sets the seed for sample_step.\n    "
        self.random_seed = random_seed
        self.rnn_cell = rnn_cell
        self.data_encoder = data_encoder
        self.latent_encoder = latent_encoder
        self.encoded_z_size = latent_encoder.output_size
        self.state_size = self.rnn_cell.state_size
        self._transition = transition
        self._emission = emission

    def zero_state(self, batch_size, dtype):
        if False:
            i = 10
            return i + 15
        'The initial state of the SRNN.\n\n    Contains the initial state of the RNN and the inital encoded latent.\n\n    Args:\n      batch_size: The batch size.\n      dtype: The data type of the SRNN.\n    Returns:\n      zero_state: The initial state of the SRNN.\n    '
        return SRNNState(rnn_state=self.rnn_cell.zero_state(batch_size, dtype), latent_encoded=tf.zeros([batch_size, self.latent_encoder.output_size], dtype=dtype))

    def run_rnn(self, prev_rnn_state, inputs):
        if False:
            return 10
        'Runs the deterministic RNN for one step.\n\n    Args:\n      prev_rnn_state: The state of the RNN from the previous timestep.\n      inputs: A Tensor of shape [batch_size, data_size], the current inputs to\n        the model. Most often this is x_{t-1}, the previous token in the\n        observation sequence.\n    Returns:\n      rnn_out: The output of the RNN.\n      rnn_state: The new state of the RNN.\n    '
        rnn_inputs = self.data_encoder(tf.to_float(inputs))
        (rnn_out, rnn_state) = self.rnn_cell(rnn_inputs, prev_rnn_state)
        return (rnn_out, rnn_state)

    def transition(self, rnn_out, prev_latent_encoded):
        if False:
            for i in range(10):
                print('nop')
        'Computes the transition distribution p(z_t|h_t, z_{t-1}).\n\n    Note that p(z_t | h_t, z_{t-1}) = p(z_t| z_{t-1}, x_{1:t-1})\n\n    Args:\n      rnn_out: The output of the rnn for the current timestep.\n      prev_latent_encoded: Float Tensor of shape\n        [batch_size, encoded_latent_size], the previous latent state z_{t-1}\n        run through latent_encoder.\n    Returns:\n      p(z_t | h_t): A normal distribution with event shape\n        [batch_size, latent_size].\n    '
        return self._transition(rnn_out, prev_latent_encoded)

    def emission(self, latent, rnn_out):
        if False:
            i = 10
            return i + 15
        'Computes the emission distribution p(x_t | z_t, h_t).\n\n    Note that p(x_t | z_t, h_t) = p(x_t | z_t, x_{1:t-1})\n\n    Args:\n      latent: The stochastic latent state z_t.\n      rnn_out: The output of the rnn for the current timestep.\n    Returns:\n      p(x_t | z_t, h_t): A distribution with event shape\n        [batch_size, data_size].\n      latent_encoded: The latent state encoded with latent_encoder. Should be\n        passed to transition() on the next timestep.\n    '
        latent_encoded = self.latent_encoder(latent)
        return (self._emission(latent_encoded, rnn_out), latent_encoded)

    def sample_step(self, prev_state, inputs, unused_t):
        if False:
            i = 10
            return i + 15
        'Samples one output from the model.\n\n    Args:\n      prev_state: The previous state of the model, a SRNNState containing the\n        previous rnn state and the previous encoded latent.\n      inputs: A Tensor of shape [batch_size, data_size], the current inputs to\n        the model. Most often this is x_{t-1}, the previous token in the\n        observation sequence.\n      unused_t: The current timestep. Not used currently.\n    Returns:\n      new_state: The next state of the model, a SRNNState.\n      xt: A float Tensor of shape [batch_size, data_size], an output sampled\n        from the emission distribution.\n    '
        (rnn_out, rnn_state) = self.run_rnn(prev_state.rnn_state, inputs)
        p_zt = self.transition(rnn_out, prev_state.latent_encoded)
        zt = p_zt.sample(seed=self.random_seed)
        (p_xt_given_zt, latent_encoded) = self.emission(zt, rnn_out)
        xt = p_xt_given_zt.sample(seed=self.random_seed)
        new_state = SRNNState(rnn_state=rnn_state, latent_encoded=latent_encoded)
        return (new_state, tf.to_float(xt))
TrainableSRNNState = namedtuple('TrainableSRNNState', SRNNState._fields + ('rnn_out',))

class TrainableSRNN(SRNN, base.ELBOTrainableSequenceModel):
    """A SRNN subclass with proposals and methods for training and evaluation.

  This class adds proposals used for training with importance-sampling based
  methods such as the ELBO. The model can be configured to propose from one
  of three proposals: a learned filtering proposal, a learned smoothing
  proposal, or the prior (i.e. the transition distribution).

  As described in the SRNN paper, the learned filtering proposal is
  parameterized by a fully connected neural network that accepts as input the
  current target x_t and the current rnn output h_t. The learned smoothing
  proposal is also given the hidden state of an RNN run in reverse over the
  inputs, so as to incorporate information about future observations.

  All learned proposals use the 'res_q' parameterization, meaning that instead
  of directly producing the mean of z_t, the proposal network predicts the
  'residual' from the prior's mean. This is explored more in section 3.3 of
  https://arxiv.org/pdf/1605.07571.pdf.

  During training, the latent state z_t is sampled from the proposal and the
  reparameterization trick is used to provide low-variance gradients.

  Note that the SRNN paper refers to the proposals as the approximate posterior,
  but we match the VRNN convention of referring to it as the encoder.
  """

    def __init__(self, rnn_cell, data_encoder, latent_encoder, transition, emission, proposal_type, proposal=None, rev_rnn_cell=None, tilt=None, random_seed=None):
        if False:
            i = 10
            return i + 15
        'Create a trainable RNN.\n\n    Args:\n      rnn_cell: A subclass of tf.nn.rnn_cell.RNNCell that will form the\n        deterministic backbone of the SRNN. The inputs to the RNN will be the\n        the encoded input of the current timestep, a Tensor of shape\n        [batch_size, encoded_data_size].\n      data_encoder: A callable that accepts a batch of data x_t and\n        \'encodes\' it, e.g. runs it through a fully connected network. Must\n        accept as argument the inputs x_t, a Tensor of the shape\n        [batch_size, data_size] and return a Tensor of shape\n        [batch_size, encoded_data_size]. This callable will be called multiple\n        times in the SRNN cell so if scoping is not handled correctly then\n        multiple copies of the variables in this network could be made. It is\n        recommended to use a snt.nets.MLP module, which takes care of this for\n        you.\n      latent_encoder: A callable that accepts a latent state z_t and\n        \'encodes\' it, e.g. runs it through a fully connected network. Must\n        accept as argument a Tensor of shape [batch_size, latent_size] and\n        return a Tensor of shape [batch_size, encoded_latent_size].\n        This callable must also have the property \'output_size\' defined,\n        returning encoded_latent_size.\n      transition: A callable that implements the transition distribution\n        p(z_t|h_t, z_t-1). Must accept as argument the previous RNN hidden state\n        and previous encoded latent state then return a tf.distributions.Normal\n        distribution conditioned on the input.\n      emission: A callable that implements the emission distribution\n        p(x_t|z_t, h_t). Must accept as arguments the encoded latent state\n        and the RNN hidden state and return a subclass of\n        tf.distributions.Distribution that can be used to evaluate the logprob\n        of the targets.\n      proposal_type: A string indicating the type of proposal to use. Can\n        be either "filtering", "smoothing", or "prior". When proposal_type is\n        "filtering" or "smoothing", proposal must be provided. When\n        proposal_type is "smoothing", rev_rnn_cell must also be provided.\n      proposal: A callable that implements the proposal q(z_t| h_t, x_{1:T}).\n        If proposal_type is "filtering" then proposal must accept as arguments\n        the current rnn output, the encoded target of the current timestep,\n        and the mean of the prior. If proposal_type is "smoothing" then\n        in addition to the current rnn output and the mean of the prior\n        proposal must accept as arguments the output of the reverse rnn.\n        proposal should return a tf.distributions.Normal distribution\n        conditioned on its inputs. If proposal_type is "prior" this argument is\n        ignored.\n      rev_rnn_cell: A subclass of tf.nn.rnn_cell.RNNCell that will aggregate\n        forward rnn outputs in the reverse direction. The inputs to the RNN\n        will be the encoded reverse input of the current timestep, a Tensor of\n        shape [batch_size, encoded_data_size].\n      tilt: A callable that implements the log of a positive tilting function\n        (ideally approximating log p(x_{t+1}|z_t, h_t). Must accept as arguments\n        the encoded latent state and the RNN hidden state and return a subclass\n        of tf.distributions.Distribution that can be used to evaluate the\n        logprob of x_{t+1}. Optionally, None and then no tilt is used.\n      random_seed: The seed for the random ops. Sets the seed for sample_step\n        and __call__.\n    '
        super(TrainableSRNN, self).__init__(rnn_cell, data_encoder, latent_encoder, transition, emission, random_seed=random_seed)
        self.rev_rnn_cell = rev_rnn_cell
        self._tilt = tilt
        assert proposal_type in ['filtering', 'smoothing', 'prior']
        self._proposal = proposal
        self.proposal_type = proposal_type
        if proposal_type != 'prior':
            assert proposal, 'If not proposing from the prior, must provide proposal.'
        if proposal_type == 'smoothing':
            assert rev_rnn_cell, 'Must provide rev_rnn_cell for smoothing proposal.'

    def zero_state(self, batch_size, dtype):
        if False:
            return 10
        super_state = super(TrainableSRNN, self).zero_state(batch_size, dtype)
        return TrainableSRNNState(rnn_out=tf.zeros([batch_size, self.rnn_cell.output_size], dtype=dtype), **super_state._asdict())

    def set_observations(self, observations, seq_lengths):
        if False:
            print('Hello World!')
        "Stores the model's observations.\n\n    Stores the observations (inputs and targets) in TensorArrays and precomputes\n    things for later like the reverse RNN output and encoded targets.\n\n    Args:\n      observations: The observations of the model, a tuple containing two\n        Tensors of shape [max_seq_len, batch_size, data_size]. The Tensors\n        should be the inputs and targets, respectively.\n      seq_lengths: An int Tensor of shape [batch_size] containing the length\n        of each sequence in observations.\n    "
        (inputs, targets) = observations
        self.seq_lengths = seq_lengths
        self.max_seq_len = tf.reduce_max(seq_lengths)
        self.targets_ta = base.ta_for_tensor(targets, clear_after_read=False)
        targets_encoded = base.encode_all(targets, self.data_encoder)
        self.targets_encoded_ta = base.ta_for_tensor(targets_encoded, clear_after_read=False)
        inputs_encoded = base.encode_all(inputs, self.data_encoder)
        (rnn_out, _) = tf.nn.dynamic_rnn(self.rnn_cell, inputs_encoded, time_major=True, dtype=tf.float32, scope='forward_rnn')
        self.rnn_ta = base.ta_for_tensor(rnn_out, clear_after_read=False)
        if self.rev_rnn_cell:
            targets_and_rnn_out = tf.concat([rnn_out, targets_encoded], 2)
            reversed_targets_and_rnn_out = tf.reverse_sequence(targets_and_rnn_out, seq_lengths, seq_axis=0, batch_axis=1)
            (reverse_rnn_out, _) = tf.nn.dynamic_rnn(self.rev_rnn_cell, reversed_targets_and_rnn_out, time_major=True, dtype=tf.float32, scope='reverse_rnn')
            reverse_rnn_out = tf.reverse_sequence(reverse_rnn_out, seq_lengths, seq_axis=0, batch_axis=1)
            self.reverse_rnn_ta = base.ta_for_tensor(reverse_rnn_out, clear_after_read=False)

    def _filtering_proposal(self, rnn_out, prev_latent_encoded, prior, t):
        if False:
            while True:
                i = 10
        'Computes the filtering proposal distribution.'
        return self._proposal(rnn_out, prev_latent_encoded, self.targets_encoded_ta.read(t), prior_mu=prior.mean())

    def _smoothing_proposal(self, rnn_out, prev_latent_encoded, prior, t):
        if False:
            while True:
                i = 10
        'Computes the smoothing proposal distribution.'
        return self._proposal(rnn_out, prev_latent_encoded, smoothing_tensors=[self.reverse_rnn_ta.read(t)], prior_mu=prior.mean())

    def proposal(self, rnn_out, prev_latent_encoded, prior, t):
        if False:
            i = 10
            return i + 15
        "Computes the proposal distribution specified by proposal_type.\n\n    Args:\n      rnn_out: The output of the rnn for the current timestep.\n      prev_latent_encoded: Float Tensor of shape\n        [batch_size, encoded_latent_size], the previous latent state z_{t-1}\n        run through latent_encoder.\n      prior: A tf.distributions.Normal distribution representing the prior\n        over z_t, p(z_t | z_{1:t-1}, x_{1:t-1}). Used for 'res_q'.\n      t: A scalar int Tensor, the current timestep.\n    "
        if self.proposal_type == 'filtering':
            return self._filtering_proposal(rnn_out, prev_latent_encoded, prior, t)
        elif self.proposal_type == 'smoothing':
            return self._smoothing_proposal(rnn_out, prev_latent_encoded, prior, t)
        elif self.proposal_type == 'prior':
            return self.transition(rnn_out, prev_latent_encoded)

    def tilt(self, rnn_out, latent_encoded, targets):
        if False:
            i = 10
            return i + 15
        r_func = self._tilt(rnn_out, latent_encoded)
        return tf.reduce_sum(r_func.log_prob(targets), axis=-1)

    def propose_and_weight(self, state, t):
        if False:
            for i in range(10):
                print('nop')
        'Runs the model and computes importance weights for one timestep.\n\n    Runs the model and computes importance weights, sampling from the proposal\n    instead of the transition/prior.\n\n    Args:\n      state: The previous state of the model, a TrainableSRNNState containing\n        the previous rnn state, the previous rnn outs, and the previous encoded\n        latent.\n      t: A scalar integer Tensor, the current timestep.\n    Returns:\n      weights: A float Tensor of shape [batch_size].\n      new_state: The new state of the model.\n    '
        targets = self.targets_ta.read(t)
        rnn_out = self.rnn_ta.read(t)
        p_zt = self.transition(rnn_out, state.latent_encoded)
        q_zt = self.proposal(rnn_out, state.latent_encoded, p_zt, t)
        zt = q_zt.sample(seed=self.random_seed)
        (p_xt_given_zt, latent_encoded) = self.emission(zt, rnn_out)
        log_p_xt_given_zt = tf.reduce_sum(p_xt_given_zt.log_prob(targets), axis=-1)
        log_p_zt = tf.reduce_sum(p_zt.log_prob(zt), axis=-1)
        log_q_zt = tf.reduce_sum(q_zt.log_prob(zt), axis=-1)
        weights = log_p_zt + log_p_xt_given_zt - log_q_zt
        if self._tilt:
            prev_log_r = tf.cond(tf.greater(t, 0), lambda : self.tilt(state.rnn_out, state.latent_encoded, targets), lambda : 0.0)
            log_r = tf.cond(tf.less(t + 1, self.max_seq_len), lambda : self.tilt(rnn_out, latent_encoded, self.targets_ta.read(t + 1)), lambda : 0.0)
            log_r *= tf.to_float(t < self.seq_lengths - 1)
            weights += log_r - prev_log_r
        rnn_out = tf.reshape(rnn_out, tf.shape(state.rnn_out))
        new_state = TrainableSRNNState(rnn_out=rnn_out, rnn_state=state.rnn_state, latent_encoded=latent_encoded)
        return (weights, new_state)
_DEFAULT_INITIALIZERS = {'w': tf.contrib.layers.xavier_initializer(), 'b': tf.zeros_initializer()}

def create_srnn(data_size, latent_size, emission_class, rnn_hidden_size=None, fcnet_hidden_sizes=None, encoded_data_size=None, encoded_latent_size=None, sigma_min=0.0, raw_sigma_bias=0.25, emission_bias_init=0.0, use_tilt=False, proposal_type='filtering', initializers=None, random_seed=None):
    if False:
        i = 10
        return i + 15
    'A factory method for creating SRNN cells.\n\n  Args:\n    data_size: The dimension of the vectors that make up the data sequences.\n    latent_size: The size of the stochastic latent state of the SRNN.\n    emission_class: The class of the emission distribution. Can be either\n      ConditionalNormalDistribution or ConditionalBernoulliDistribution.\n    rnn_hidden_size: The hidden state dimension of the RNN that forms the\n      deterministic part of this SRNN. If None, then it defaults\n      to latent_size.\n    fcnet_hidden_sizes: A list of python integers, the size of the hidden\n      layers of the fully connected networks that parameterize the conditional\n      distributions of the SRNN. If None, then it defaults to one hidden\n      layer of size latent_size.\n    encoded_data_size: The size of the output of the data encoding network. If\n      None, defaults to latent_size.\n    encoded_latent_size: The size of the output of the latent state encoding\n      network. If None, defaults to latent_size.\n    sigma_min: The minimum value that the standard deviation of the\n      distribution over the latent state can take.\n    raw_sigma_bias: A scalar that is added to the raw standard deviation\n      output from the neural networks that parameterize the prior and\n      approximate posterior. Useful for preventing standard deviations close\n      to zero.\n    emission_bias_init: A bias to added to the raw output of the fully\n      connected network that parameterizes the emission distribution. Useful\n      for initalizing the mean of the distribution to a sensible starting point\n      such as the mean of the training data. Only used with Bernoulli generative\n      distributions.\n    use_tilt: If true, create a SRNN with a tilting function.\n    proposal_type: The type of proposal to use. Can be "filtering", "smoothing",\n      or "prior".\n    initializers: The variable intitializers to use for the fully connected\n      networks and RNN cell. Must be a dictionary mapping the keys \'w\' and \'b\'\n      to the initializers for the weights and biases. Defaults to xavier for\n      the weights and zeros for the biases when initializers is None.\n    random_seed: A random seed for the SRNN resampling operations.\n  Returns:\n    model: A TrainableSRNN object.\n  '
    if rnn_hidden_size is None:
        rnn_hidden_size = latent_size
    if fcnet_hidden_sizes is None:
        fcnet_hidden_sizes = [latent_size]
    if encoded_data_size is None:
        encoded_data_size = latent_size
    if encoded_latent_size is None:
        encoded_latent_size = latent_size
    if initializers is None:
        initializers = _DEFAULT_INITIALIZERS
    data_encoder = snt.nets.MLP(output_sizes=fcnet_hidden_sizes + [encoded_data_size], initializers=initializers, name='data_encoder')
    latent_encoder = snt.nets.MLP(output_sizes=fcnet_hidden_sizes + [encoded_latent_size], initializers=initializers, name='latent_encoder')
    transition = base.ConditionalNormalDistribution(size=latent_size, hidden_layer_sizes=fcnet_hidden_sizes, sigma_min=sigma_min, raw_sigma_bias=raw_sigma_bias, initializers=initializers, name='prior')
    if emission_class == base.ConditionalBernoulliDistribution:
        emission_dist = functools.partial(base.ConditionalBernoulliDistribution, bias_init=emission_bias_init)
    else:
        emission_dist = base.ConditionalNormalDistribution
    emission = emission_dist(size=data_size, hidden_layer_sizes=fcnet_hidden_sizes, initializers=initializers, name='generative')
    if proposal_type in ['filtering', 'smoothing']:
        proposal = base.NormalApproximatePosterior(size=latent_size, hidden_layer_sizes=fcnet_hidden_sizes, sigma_min=sigma_min, raw_sigma_bias=raw_sigma_bias, initializers=initializers, smoothing=proposal_type == 'smoothing', name='approximate_posterior')
    else:
        proposal = None
    if use_tilt:
        tilt = emission_dist(size=data_size, hidden_layer_sizes=fcnet_hidden_sizes, initializers=initializers, name='tilt')
    else:
        tilt = None
    rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden_size, initializer=initializers['w'])
    rev_rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden_size, initializer=initializers['w'])
    return TrainableSRNN(rnn_cell, data_encoder, latent_encoder, transition, emission, proposal_type, proposal=proposal, rev_rnn_cell=rev_rnn_cell, tilt=tilt, random_seed=random_seed)