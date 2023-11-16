"""Model construction."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from models import bidirectional
from models import bidirectional_vd
from models import bidirectional_zaremba
from models import cnn
from models import critic_vd
from models import feedforward
from models import rnn
from models import rnn_nas
from models import rnn_vd
from models import rnn_zaremba
from models import seq2seq
from models import seq2seq_nas
from models import seq2seq_vd
from models import seq2seq_zaremba
FLAGS = tf.app.flags.FLAGS

def create_generator(hparams, inputs, targets, present, is_training, is_validating, reuse=None):
    if False:
        print('Hello World!')
    'Create the Generator model specified by the FLAGS and hparams.\n\n  Args;\n    hparams:  Hyperparameters for the MaskGAN.\n    inputs:  tf.int32 Tensor of the sequence input of shape [batch_size,\n      sequence_length].\n    present:  tf.bool Tensor indicating the presence or absence of the token\n      of shape [batch_size, sequence_length].\n    is_training:  Whether the model is training.\n    is_validating:  Whether the model is being run in validation mode for\n      calculating the perplexity.\n    reuse (Optional):  Whether to reuse the model.\n\n  Returns:\n    Tuple of the (sequence, logits, log_probs) of the Generator.   Sequence\n      and logits have shape [batch_size, sequence_length, vocab_size].  The\n      log_probs will have shape [batch_size, sequence_length].  Log_probs\n      corresponds to the log probability of selecting the words.\n  '
    if FLAGS.generator_model == 'rnn':
        (sequence, logits, log_probs, initial_state, final_state) = rnn.generator(hparams, inputs, targets, present, is_training=is_training, is_validating=is_validating, reuse=reuse)
    elif FLAGS.generator_model == 'rnn_zaremba':
        (sequence, logits, log_probs, initial_state, final_state) = rnn_zaremba.generator(hparams, inputs, targets, present, is_training=is_training, is_validating=is_validating, reuse=reuse)
    elif FLAGS.generator_model == 'seq2seq':
        (sequence, logits, log_probs, initial_state, final_state) = seq2seq.generator(hparams, inputs, targets, present, is_training=is_training, is_validating=is_validating, reuse=reuse)
    elif FLAGS.generator_model == 'seq2seq_zaremba':
        (sequence, logits, log_probs, initial_state, final_state) = seq2seq_zaremba.generator(hparams, inputs, targets, present, is_training=is_training, is_validating=is_validating, reuse=reuse)
    elif FLAGS.generator_model == 'rnn_nas':
        (sequence, logits, log_probs, initial_state, final_state) = rnn_nas.generator(hparams, inputs, targets, present, is_training=is_training, is_validating=is_validating, reuse=reuse)
    elif FLAGS.generator_model == 'seq2seq_nas':
        (sequence, logits, log_probs, initial_state, final_state) = seq2seq_nas.generator(hparams, inputs, targets, present, is_training=is_training, is_validating=is_validating, reuse=reuse)
    elif FLAGS.generator_model == 'seq2seq_vd':
        (sequence, logits, log_probs, initial_state, final_state, encoder_states) = seq2seq_vd.generator(hparams, inputs, targets, present, is_training=is_training, is_validating=is_validating, reuse=reuse)
    else:
        raise NotImplementedError
    return (sequence, logits, log_probs, initial_state, final_state, encoder_states)

def create_discriminator(hparams, sequence, is_training, reuse=None, initial_state=None, inputs=None, present=None):
    if False:
        return 10
    'Create the Discriminator model specified by the FLAGS and hparams.\n\n  Args:\n    hparams:  Hyperparameters for the MaskGAN.\n    sequence:  tf.int32 Tensor sequence of shape [batch_size, sequence_length]\n    is_training:  Whether the model is training.\n    reuse (Optional):  Whether to reuse the model.\n\n  Returns:\n    predictions:  tf.float32 Tensor of predictions of shape [batch_size,\n      sequence_length]\n  '
    if FLAGS.discriminator_model == 'cnn':
        predictions = cnn.discriminator(hparams, sequence, is_training=is_training, reuse=reuse)
    elif FLAGS.discriminator_model == 'fnn':
        predictions = feedforward.discriminator(hparams, sequence, is_training=is_training, reuse=reuse)
    elif FLAGS.discriminator_model == 'rnn':
        predictions = rnn.discriminator(hparams, sequence, is_training=is_training, reuse=reuse)
    elif FLAGS.discriminator_model == 'bidirectional':
        predictions = bidirectional.discriminator(hparams, sequence, is_training=is_training, reuse=reuse)
    elif FLAGS.discriminator_model == 'bidirectional_zaremba':
        predictions = bidirectional_zaremba.discriminator(hparams, sequence, is_training=is_training, reuse=reuse)
    elif FLAGS.discriminator_model == 'seq2seq_vd':
        predictions = seq2seq_vd.discriminator(hparams, inputs, present, sequence, is_training=is_training, reuse=reuse)
    elif FLAGS.discriminator_model == 'rnn_zaremba':
        predictions = rnn_zaremba.discriminator(hparams, sequence, is_training=is_training, reuse=reuse)
    elif FLAGS.discriminator_model == 'rnn_nas':
        predictions = rnn_nas.discriminator(hparams, sequence, is_training=is_training, reuse=reuse)
    elif FLAGS.discriminator_model == 'rnn_vd':
        predictions = rnn_vd.discriminator(hparams, sequence, is_training=is_training, reuse=reuse, initial_state=initial_state)
    elif FLAGS.discriminator_model == 'bidirectional_vd':
        predictions = bidirectional_vd.discriminator(hparams, sequence, is_training=is_training, reuse=reuse, initial_state=initial_state)
    else:
        raise NotImplementedError
    return predictions

def create_critic(hparams, sequence, is_training, reuse=None):
    if False:
        while True:
            i = 10
    'Create the Critic model specified by the FLAGS and hparams.\n\n  Args:\n    hparams:  Hyperparameters for the MaskGAN.\n    sequence:  tf.int32 Tensor sequence of shape [batch_size, sequence_length]\n    is_training:  Whether the model is training.\n    reuse (Optional):  Whether to reuse the model.\n\n  Returns:\n    values:  tf.float32 Tensor of predictions of shape [batch_size,\n      sequence_length]\n  '
    if FLAGS.baseline_method == 'critic':
        if FLAGS.discriminator_model == 'seq2seq_vd':
            values = critic_vd.critic_seq2seq_vd_derivative(hparams, sequence, is_training, reuse=reuse)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return values