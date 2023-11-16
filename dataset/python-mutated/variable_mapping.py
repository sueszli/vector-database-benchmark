from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

def rnn_nas(hparams, model):
    if False:
        while True:
            i = 10
    assert model == 'gen' or model == 'dis'
    if model == 'gen':
        assert FLAGS.generator_model == 'rnn_nas'
        assert hparams.gen_num_layers == 2
    if model == 'dis':
        assert FLAGS.discriminator_model == 'rnn_nas'
        assert hparams.dis_num_layers == 2
    if model == 'gen':
        softmax_b = [v for v in tf.trainable_variables() if v.op.name == 'gen/rnn/softmax_b'][0]
    embedding = [v for v in tf.trainable_variables() if v.op.name == str(model) + '/rnn/embedding'][0]
    lstm_w_0 = [v for v in tf.trainable_variables() if v.op.name == str(model) + '/rnn/GenericMultiRNNCell/Cell0/Alien/rnn_builder/big_h_mat'][0]
    lstm_b_0 = [v for v in tf.trainable_variables() if v.op.name == str(model) + '/rnn/GenericMultiRNNCell/Cell0/Alien/rnn_builder/big_inputs_mat'][0]
    lstm_w_1 = [v for v in tf.trainable_variables() if v.op.name == str(model) + '/rnn/GenericMultiRNNCell/Cell1/Alien/rnn_builder/big_h_mat'][0]
    lstm_b_1 = [v for v in tf.trainable_variables() if v.op.name == str(model) + '/rnn/GenericMultiRNNCell/Cell1/Alien/rnn_builder/big_inputs_mat'][0]
    if model == 'gen':
        variable_mapping = {'Model/embeddings/input_embedding': embedding, 'Model/RNN/GenericMultiRNNCell/Cell0/Alien/rnn_builder/big_h_mat': lstm_w_0, 'Model/RNN/GenericMultiRNNCell/Cell0/Alien/rnn_builder/big_inputs_mat': lstm_b_0, 'Model/RNN/GenericMultiRNNCell/Cell1/Alien/rnn_builder/big_h_mat': lstm_w_1, 'Model/RNN/GenericMultiRNNCell/Cell1/Alien/rnn_builder/big_inputs_mat': lstm_b_1, 'Model/softmax_b': softmax_b}
    else:
        variable_mapping = {'Model/embeddings/input_embedding': embedding, 'Model/RNN/GenericMultiRNNCell/Cell0/Alien/rnn_builder/big_h_mat': lstm_w_0, 'Model/RNN/GenericMultiRNNCell/Cell0/Alien/rnn_builder/big_inputs_mat': lstm_b_0, 'Model/RNN/GenericMultiRNNCell/Cell1/Alien/rnn_builder/big_h_mat': lstm_w_1, 'Model/RNN/GenericMultiRNNCell/Cell1/Alien/rnn_builder/big_inputs_mat': lstm_b_1}
    return variable_mapping

def cnn():
    if False:
        for i in range(10):
            print('nop')
    'Variable mapping for the CNN embedding.\n\n  Returns:\n    variable_mapping:  Dictionary with Key: ckpt_name, Value: model_var.\n  '
    assert FLAGS.discriminator_model == 'cnn'
    embedding = [v for v in tf.trainable_variables() if v.op.name == 'dis/embedding'][0]
    variable_mapping = {'Model/embedding': embedding}
    return variable_mapping

def rnn_zaremba(hparams, model):
    if False:
        i = 10
        return i + 15
    "Returns the PTB Variable name to MaskGAN Variable dictionary mapping.  This\n  is a highly restrictive function just for testing.  This will need to be\n  generalized.\n\n  Args:\n    hparams:  Hyperparameters for the MaskGAN.\n    model:  Model type, one of ['gen', 'dis'].\n\n  Returns:\n    variable_mapping:  Dictionary with Key: ckpt_name, Value: model_var.\n  "
    assert model == 'gen' or model == 'dis'
    if model == 'gen':
        assert FLAGS.generator_model == 'rnn_zaremba'
        assert hparams.gen_num_layers == 2
    if model == 'dis':
        assert FLAGS.discriminator_model == 'rnn_zaremba' or FLAGS.discriminator_model == 'rnn_vd'
        assert hparams.dis_num_layers == 2
    if model == 'gen':
        softmax_w = [v for v in tf.trainable_variables() if v.op.name == 'gen/rnn/softmax_w'][0]
        softmax_b = [v for v in tf.trainable_variables() if v.op.name == 'gen/rnn/softmax_b'][0]
    if not FLAGS.dis_share_embedding or model != 'dis':
        embedding = [v for v in tf.trainable_variables() if v.op.name == str(model) + '/rnn/embedding'][0]
    lstm_w_0 = [v for v in tf.trainable_variables() if v.op.name == str(model) + '/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'][0]
    lstm_b_0 = [v for v in tf.trainable_variables() if v.op.name == str(model) + '/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'][0]
    lstm_w_1 = [v for v in tf.trainable_variables() if v.op.name == str(model) + '/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'][0]
    lstm_b_1 = [v for v in tf.trainable_variables() if v.op.name == str(model) + '/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias'][0]
    if model == 'gen':
        variable_mapping = {'Model/embedding': embedding, 'Model/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/kernel': lstm_w_0, 'Model/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/bias': lstm_b_0, 'Model/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/kernel': lstm_w_1, 'Model/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/bias': lstm_b_1, 'Model/softmax_w': softmax_w, 'Model/softmax_b': softmax_b}
    elif FLAGS.dis_share_embedding:
        variable_mapping = {'Model/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/kernel': lstm_w_0, 'Model/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/bias': lstm_b_0, 'Model/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/kernel': lstm_w_1, 'Model/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/bias': lstm_b_1}
    else:
        variable_mapping = {'Model/embedding': embedding, 'Model/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/kernel': lstm_w_0, 'Model/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/bias': lstm_b_0, 'Model/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/kernel': lstm_w_1, 'Model/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/bias': lstm_b_1}
    return variable_mapping

def gen_encoder_seq2seq_nas(hparams):
    if False:
        i = 10
        return i + 15
    'Returns the NAS Variable name to MaskGAN Variable\n  dictionary mapping.  This is a highly restrictive function just for testing.\n  This is for the *unidirecitional* seq2seq_nas encoder.\n\n  Args:\n    hparams:  Hyperparameters for the MaskGAN.\n\n  Returns:\n    variable_mapping:  Dictionary with Key: ckpt_name, Value: model_varself.\n  '
    assert FLAGS.generator_model == 'seq2seq_nas'
    assert hparams.gen_num_layers == 2
    if not FLAGS.seq2seq_share_embedding:
        encoder_embedding = [v for v in tf.trainable_variables() if v.op.name == 'gen/encoder/rnn/embedding'][0]
    encoder_lstm_w_0 = [v for v in tf.trainable_variables() if v.op.name == 'gen/encoder/rnn/GenericMultiRNNCell/Cell0/Alien/rnn_builder/big_h_mat'][0]
    encoder_lstm_b_0 = [v for v in tf.trainable_variables() if v.op.name == 'gen/encoder/rnn/GenericMultiRNNCell/Cell0/Alien/rnn_builder/big_inputs_mat'][0]
    encoder_lstm_w_1 = [v for v in tf.trainable_variables() if v.op.name == 'gen/encoder/rnn/GenericMultiRNNCell/Cell1/Alien/rnn_builder/big_h_mat'][0]
    encoder_lstm_b_1 = [v for v in tf.trainable_variables() if v.op.name == 'gen/encoder/rnn/GenericMultiRNNCell/Cell1/Alien/rnn_builder/big_inputs_mat'][0]
    if not FLAGS.seq2seq_share_embedding:
        variable_mapping = {'Model/embeddings/input_embedding': encoder_embedding, 'Model/RNN/GenericMultiRNNCell/Cell0/Alien/rnn_builder/big_h_mat': encoder_lstm_w_0, 'Model/RNN/GenericMultiRNNCell/Cell0/Alien/rnn_builder/big_inputs_mat': encoder_lstm_b_0, 'Model/RNN/GenericMultiRNNCell/Cell1/Alien/rnn_builder/big_h_mat': encoder_lstm_w_1, 'Model/RNN/GenericMultiRNNCell/Cell1/Alien/rnn_builder/big_inputs_mat': encoder_lstm_b_1}
    else:
        variable_mapping = {'Model/RNN/GenericMultiRNNCell/Cell0/Alien/rnn_builder/big_h_mat': encoder_lstm_w_0, 'Model/RNN/GenericMultiRNNCell/Cell0/Alien/rnn_builder/big_inputs_mat': encoder_lstm_b_0, 'Model/RNN/GenericMultiRNNCell/Cell1/Alien/rnn_builder/big_h_mat': encoder_lstm_w_1, 'Model/RNN/GenericMultiRNNCell/Cell1/Alien/rnn_builder/big_inputs_mat': encoder_lstm_b_1}
    return variable_mapping

def gen_decoder_seq2seq_nas(hparams):
    if False:
        while True:
            i = 10
    assert FLAGS.generator_model == 'seq2seq_nas'
    assert hparams.gen_num_layers == 2
    decoder_embedding = [v for v in tf.trainable_variables() if v.op.name == 'gen/decoder/rnn/embedding'][0]
    decoder_lstm_w_0 = [v for v in tf.trainable_variables() if v.op.name == 'gen/decoder/rnn/GenericMultiRNNCell/Cell0/Alien/rnn_builder/big_h_mat'][0]
    decoder_lstm_b_0 = [v for v in tf.trainable_variables() if v.op.name == 'gen/decoder/rnn/GenericMultiRNNCell/Cell0/Alien/rnn_builder/big_inputs_mat'][0]
    decoder_lstm_w_1 = [v for v in tf.trainable_variables() if v.op.name == 'gen/decoder/rnn/GenericMultiRNNCell/Cell1/Alien/rnn_builder/big_h_mat'][0]
    decoder_lstm_b_1 = [v for v in tf.trainable_variables() if v.op.name == 'gen/decoder/rnn/GenericMultiRNNCell/Cell1/Alien/rnn_builder/big_inputs_mat'][0]
    decoder_softmax_b = [v for v in tf.trainable_variables() if v.op.name == 'gen/decoder/rnn/softmax_b'][0]
    variable_mapping = {'Model/embeddings/input_embedding': decoder_embedding, 'Model/RNN/GenericMultiRNNCell/Cell0/Alien/rnn_builder/big_h_mat': decoder_lstm_w_0, 'Model/RNN/GenericMultiRNNCell/Cell0/Alien/rnn_builder/big_inputs_mat': decoder_lstm_b_0, 'Model/RNN/GenericMultiRNNCell/Cell1/Alien/rnn_builder/big_h_mat': decoder_lstm_w_1, 'Model/RNN/GenericMultiRNNCell/Cell1/Alien/rnn_builder/big_inputs_mat': decoder_lstm_b_1, 'Model/softmax_b': decoder_softmax_b}
    return variable_mapping

def gen_encoder_seq2seq(hparams):
    if False:
        return 10
    'Returns the PTB Variable name to MaskGAN Variable\n  dictionary mapping.  This is a highly restrictive function just for testing.\n  This is foe the *unidirecitional* seq2seq_zaremba encoder.\n\n  Args:\n    hparams:  Hyperparameters for the MaskGAN.\n\n  Returns:\n    variable_mapping:  Dictionary with Key: ckpt_name, Value: model_varself.\n  '
    assert FLAGS.generator_model == 'seq2seq_zaremba' or FLAGS.generator_model == 'seq2seq_vd'
    assert hparams.gen_num_layers == 2
    if not FLAGS.seq2seq_share_embedding:
        encoder_embedding = [v for v in tf.trainable_variables() if v.op.name == 'gen/encoder/rnn/embedding'][0]
    encoder_lstm_w_0 = [v for v in tf.trainable_variables() if v.op.name == 'gen/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'][0]
    encoder_lstm_b_0 = [v for v in tf.trainable_variables() if v.op.name == 'gen/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'][0]
    encoder_lstm_w_1 = [v for v in tf.trainable_variables() if v.op.name == 'gen/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'][0]
    encoder_lstm_b_1 = [v for v in tf.trainable_variables() if v.op.name == 'gen/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias'][0]
    if FLAGS.data_set == 'ptb':
        model_str = 'Model'
    else:
        model_str = 'model'
    if not FLAGS.seq2seq_share_embedding:
        variable_mapping = {str(model_str) + '/embedding': encoder_embedding, str(model_str) + '/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/kernel': encoder_lstm_w_0, str(model_str) + '/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/bias': encoder_lstm_b_0, str(model_str) + '/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/kernel': encoder_lstm_w_1, str(model_str) + '/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/bias': encoder_lstm_b_1}
    else:
        variable_mapping = {str(model_str) + '/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/kernel': encoder_lstm_w_0, str(model_str) + '/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/bias': encoder_lstm_b_0, str(model_str) + '/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/kernel': encoder_lstm_w_1, str(model_str) + '/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/bias': encoder_lstm_b_1}
    return variable_mapping

def gen_decoder_seq2seq(hparams):
    if False:
        while True:
            i = 10
    assert FLAGS.generator_model == 'seq2seq_zaremba' or FLAGS.generator_model == 'seq2seq_vd'
    assert hparams.gen_num_layers == 2
    decoder_embedding = [v for v in tf.trainable_variables() if v.op.name == 'gen/decoder/rnn/embedding'][0]
    decoder_lstm_w_0 = [v for v in tf.trainable_variables() if v.op.name == 'gen/decoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'][0]
    decoder_lstm_b_0 = [v for v in tf.trainable_variables() if v.op.name == 'gen/decoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'][0]
    decoder_lstm_w_1 = [v for v in tf.trainable_variables() if v.op.name == 'gen/decoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'][0]
    decoder_lstm_b_1 = [v for v in tf.trainable_variables() if v.op.name == 'gen/decoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias'][0]
    decoder_softmax_b = [v for v in tf.trainable_variables() if v.op.name == 'gen/decoder/rnn/softmax_b'][0]
    if FLAGS.data_set == 'ptb':
        model_str = 'Model'
    else:
        model_str = 'model'
    variable_mapping = {str(model_str) + '/embedding': decoder_embedding, str(model_str) + '/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/kernel': decoder_lstm_w_0, str(model_str) + '/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/bias': decoder_lstm_b_0, str(model_str) + '/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/kernel': decoder_lstm_w_1, str(model_str) + '/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/bias': decoder_lstm_b_1, str(model_str) + '/softmax_b': decoder_softmax_b}
    return variable_mapping

def dis_fwd_bidirectional(hparams):
    if False:
        print('Hello World!')
    'Returns the *forward* PTB Variable name to MaskGAN Variable dictionary\n  mapping.  This is a highly restrictive function just for testing. This is for\n  the bidirectional_zaremba discriminator.\n\n  Args:\n    FLAGS:  Flags for the model.\n    hparams:  Hyperparameters for the MaskGAN.\n\n  Returns:\n    variable_mapping:  Dictionary with Key: ckpt_name, Value: model_varself.\n  '
    assert FLAGS.discriminator_model == 'bidirectional_zaremba' or FLAGS.discriminator_model == 'bidirectional_vd'
    assert hparams.dis_num_layers == 2
    if not FLAGS.dis_share_embedding:
        embedding = [v for v in tf.trainable_variables() if v.op.name == 'dis/embedding'][0]
    fw_lstm_w_0 = [v for v in tf.trainable_variables() if v.op.name == 'dis/rnn/fw/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'][0]
    fw_lstm_b_0 = [v for v in tf.trainable_variables() if v.op.name == 'dis/rnn/fw/multi_rnn_cell/cell_0/basic_lstm_cell/bias'][0]
    fw_lstm_w_1 = [v for v in tf.trainable_variables() if v.op.name == 'dis/rnn/fw/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'][0]
    fw_lstm_b_1 = [v for v in tf.trainable_variables() if v.op.name == 'dis/rnn/fw/multi_rnn_cell/cell_1/basic_lstm_cell/bias'][0]
    if FLAGS.dis_share_embedding:
        variable_mapping = {'Model/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/kernel': fw_lstm_w_0, 'Model/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/bias': fw_lstm_b_0, 'Model/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/kernel': fw_lstm_w_1, 'Model/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/bias': fw_lstm_b_1}
    else:
        variable_mapping = {'Model/embedding': embedding, 'Model/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/kernel': fw_lstm_w_0, 'Model/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/bias': fw_lstm_b_0, 'Model/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/kernel': fw_lstm_w_1, 'Model/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/bias': fw_lstm_b_1}
    return variable_mapping

def dis_bwd_bidirectional(hparams):
    if False:
        print('Hello World!')
    'Returns the *backward* PTB Variable name to MaskGAN Variable dictionary\n  mapping.  This is a highly restrictive function just for testing. This is for\n  the bidirectional_zaremba discriminator.\n\n  Args:\n    hparams:  Hyperparameters for the MaskGAN.\n\n  Returns:\n    variable_mapping:  Dictionary with Key: ckpt_name, Value: model_varself.\n  '
    assert FLAGS.discriminator_model == 'bidirectional_zaremba' or FLAGS.discriminator_model == 'bidirectional_vd'
    assert hparams.dis_num_layers == 2
    bw_lstm_w_0 = [v for v in tf.trainable_variables() if v.op.name == 'dis/rnn/bw/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'][0]
    bw_lstm_b_0 = [v for v in tf.trainable_variables() if v.op.name == 'dis/rnn/bw/multi_rnn_cell/cell_0/basic_lstm_cell/bias'][0]
    bw_lstm_w_1 = [v for v in tf.trainable_variables() if v.op.name == 'dis/rnn/bw/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'][0]
    bw_lstm_b_1 = [v for v in tf.trainable_variables() if v.op.name == 'dis/rnn/bw/multi_rnn_cell/cell_1/basic_lstm_cell/bias'][0]
    variable_mapping = {'Model/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/kernel': bw_lstm_w_0, 'Model/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/bias': bw_lstm_b_0, 'Model/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/kernel': bw_lstm_w_1, 'Model/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/bias': bw_lstm_b_1}
    return variable_mapping

def dis_encoder_seq2seq(hparams):
    if False:
        while True:
            i = 10
    'Returns the PTB Variable name to MaskGAN Variable\n  dictionary mapping.\n\n  Args:\n    hparams:  Hyperparameters for the MaskGAN.\n\n  Returns:\n    variable_mapping:  Dictionary with Key: ckpt_name, Value: model_varself.\n  '
    assert FLAGS.discriminator_model == 'seq2seq_vd'
    assert hparams.dis_num_layers == 2
    encoder_lstm_w_0 = [v for v in tf.trainable_variables() if v.op.name == 'dis/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'][0]
    encoder_lstm_b_0 = [v for v in tf.trainable_variables() if v.op.name == 'dis/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'][0]
    encoder_lstm_w_1 = [v for v in tf.trainable_variables() if v.op.name == 'dis/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'][0]
    encoder_lstm_b_1 = [v for v in tf.trainable_variables() if v.op.name == 'dis/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias'][0]
    if FLAGS.data_set == 'ptb':
        model_str = 'Model'
    else:
        model_str = 'model'
    variable_mapping = {str(model_str) + '/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/kernel': encoder_lstm_w_0, str(model_str) + '/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/bias': encoder_lstm_b_0, str(model_str) + '/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/kernel': encoder_lstm_w_1, str(model_str) + '/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/bias': encoder_lstm_b_1}
    return variable_mapping

def dis_decoder_seq2seq(hparams):
    if False:
        for i in range(10):
            print('nop')
    assert FLAGS.discriminator_model == 'seq2seq_vd'
    assert hparams.dis_num_layers == 2
    if not FLAGS.dis_share_embedding:
        decoder_embedding = [v for v in tf.trainable_variables() if v.op.name == 'dis/decoder/rnn/embedding'][0]
    decoder_lstm_w_0 = [v for v in tf.trainable_variables() if v.op.name == 'dis/decoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'][0]
    decoder_lstm_b_0 = [v for v in tf.trainable_variables() if v.op.name == 'dis/decoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'][0]
    decoder_lstm_w_1 = [v for v in tf.trainable_variables() if v.op.name == 'dis/decoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'][0]
    decoder_lstm_b_1 = [v for v in tf.trainable_variables() if v.op.name == 'dis/decoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias'][0]
    if FLAGS.data_set == 'ptb':
        model_str = 'Model'
    else:
        model_str = 'model'
    if not FLAGS.dis_share_embedding:
        variable_mapping = {str(model_str) + '/embedding': decoder_embedding, str(model_str) + '/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/kernel': decoder_lstm_w_0, str(model_str) + '/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/bias': decoder_lstm_b_0, str(model_str) + '/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/kernel': decoder_lstm_w_1, str(model_str) + '/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/bias': decoder_lstm_b_1}
    else:
        variable_mapping = {str(model_str) + '/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/kernel': decoder_lstm_w_0, str(model_str) + '/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/bias': decoder_lstm_b_0, str(model_str) + '/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/kernel': decoder_lstm_w_1, str(model_str) + '/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/bias': decoder_lstm_b_1}
    return variable_mapping

def dis_seq2seq_vd(hparams):
    if False:
        for i in range(10):
            print('nop')
    assert FLAGS.discriminator_model == 'seq2seq_vd'
    assert hparams.dis_num_layers == 2
    if not FLAGS.dis_share_embedding:
        decoder_embedding = [v for v in tf.trainable_variables() if v.op.name == 'dis/decoder/rnn/embedding'][0]
    encoder_lstm_w_0 = [v for v in tf.trainable_variables() if v.op.name == 'dis/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'][0]
    encoder_lstm_b_0 = [v for v in tf.trainable_variables() if v.op.name == 'dis/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'][0]
    encoder_lstm_w_1 = [v for v in tf.trainable_variables() if v.op.name == 'dis/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'][0]
    encoder_lstm_b_1 = [v for v in tf.trainable_variables() if v.op.name == 'dis/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias'][0]
    if FLAGS.attention_option is not None:
        decoder_attention_keys = [v for v in tf.trainable_variables() if v.op.name == 'dis/decoder/attention_keys/weights'][0]
        decoder_attention_construct_weights = [v for v in tf.trainable_variables() if v.op.name == 'dis/decoder/rnn/attention_construct/weights'][0]
    decoder_lstm_w_0 = [v for v in tf.trainable_variables() if v.op.name == 'dis/decoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'][0]
    decoder_lstm_b_0 = [v for v in tf.trainable_variables() if v.op.name == 'dis/decoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'][0]
    decoder_lstm_w_1 = [v for v in tf.trainable_variables() if v.op.name == 'dis/decoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'][0]
    decoder_lstm_b_1 = [v for v in tf.trainable_variables() if v.op.name == 'dis/decoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias'][0]
    variable_mapping = {'gen/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel': encoder_lstm_w_0, 'gen/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias': encoder_lstm_b_0, 'gen/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel': encoder_lstm_w_1, 'gen/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias': encoder_lstm_b_1, 'gen/decoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel': decoder_lstm_w_0, 'gen/decoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias': decoder_lstm_b_0, 'gen/decoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel': decoder_lstm_w_1, 'gen/decoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias': decoder_lstm_b_1}
    if not FLAGS.dis_share_embedding:
        variable_mapping['gen/decoder/rnn/embedding'] = decoder_embedding
    if FLAGS.attention_option is not None:
        variable_mapping['gen/decoder/attention_keys/weights'] = decoder_attention_keys
        variable_mapping['gen/decoder/rnn/attention_construct/weights'] = decoder_attention_construct_weights
    return variable_mapping