"""Launch example:

[IMDB]
python train_mask_gan.py --data_dir
/tmp/imdb  --data_set imdb  --batch_size 128
--sequence_length 20  --base_directory /tmp/maskGAN_v0.01
--hparams="gen_rnn_size=650,gen_num_layers=2,dis_rnn_size=650,dis_num_layers=2
,critic_learning_rate=0.0009756,dis_learning_rate=0.0000585,
dis_train_iterations=8,gen_learning_rate=0.0016624,
gen_full_learning_rate_steps=1e9,gen_learning_rate_decay=0.999999,
rl_discount_rate=0.8835659"  --mode TRAIN  --max_steps 1000000
--generator_model seq2seq_vd  --discriminator_model seq2seq_vd
--is_present_rate 0.5  --summaries_every 25  --print_every 25
 --max_num_to_print=3 --generator_optimizer=adam
 --seq2seq_share_embedding=True --baseline_method=critic
 --attention_option=luong --n_gram_eval=4 --mask_strategy=contiguous
 --gen_training_strategy=reinforce --dis_pretrain_steps=100
 --perplexity_threshold=1000000
 --dis_share_embedding=True  --maskgan_ckpt
 /tmp/model.ckpt-171091
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
from functools import partial
import os
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
import pretrain_mask_gan
from data import imdb_loader
from data import ptb_loader
from model_utils import helper
from model_utils import model_construction
from model_utils import model_losses
from model_utils import model_optimization
from model_utils import model_utils
from model_utils import n_gram
from models import evaluation_utils
from models import rollout
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
MODE_TRAIN = 'TRAIN'
MODE_TRAIN_EVAL = 'TRAIN_EVAL'
MODE_VALIDATION = 'VALIDATION'
MODE_TEST = 'TEST'
tf.app.flags.DEFINE_enum('mode', 'TRAIN', [MODE_TRAIN, MODE_VALIDATION, MODE_TEST, MODE_TRAIN_EVAL], 'What this binary will do.')
tf.app.flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
tf.app.flags.DEFINE_string('eval_master', '', 'Name prefix of the Tensorflow eval master.')
tf.app.flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')
tf.app.flags.DEFINE_integer('ps_tasks', 0, 'Number of tasks in the ps job.\n                            If 0 no ps job is used.')
tf.app.flags.DEFINE_string('hparams', '', 'Comma separated list of name=value hyperparameter pairs.')
tf.app.flags.DEFINE_integer('batch_size', 20, 'The batch size.')
tf.app.flags.DEFINE_integer('vocab_size', 10000, 'The vocabulary size.')
tf.app.flags.DEFINE_integer('sequence_length', 20, 'The sequence length.')
tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Maximum number of steps to run.')
tf.app.flags.DEFINE_string('mask_strategy', 'random', "Strategy for masking the words.  Determine the characterisitics of how the words are dropped out.  One of ['contiguous', 'random'].")
tf.app.flags.DEFINE_float('is_present_rate', 0.5, 'Percent of tokens present in the forward sequence.')
tf.app.flags.DEFINE_float('is_present_rate_decay', None, 'Decay rate for the percent of words that are real (are present).')
tf.app.flags.DEFINE_string('generator_model', 'seq2seq', "Type of Generator model.  One of ['rnn', 'seq2seq', 'seq2seq_zaremba','rnn_zaremba', 'rnn_nas', 'seq2seq_nas']")
tf.app.flags.DEFINE_string('attention_option', None, "Attention mechanism.  One of [None, 'luong', 'bahdanau']")
tf.app.flags.DEFINE_string('discriminator_model', 'bidirectional', "Type of Discriminator model.  One of ['cnn', 'rnn', 'bidirectional', 'rnn_zaremba', 'bidirectional_zaremba', 'rnn_nas', 'rnn_vd', 'seq2seq_vd']")
tf.app.flags.DEFINE_boolean('seq2seq_share_embedding', False, 'Whether to share the embeddings between the encoder and decoder.')
tf.app.flags.DEFINE_boolean('dis_share_embedding', False, 'Whether to share the embeddings between the generator and discriminator.')
tf.app.flags.DEFINE_boolean('dis_update_share_embedding', False, 'Whether the discriminator should update the shared embedding.')
tf.app.flags.DEFINE_boolean('use_gen_mode', False, 'Use the mode of the generator to produce samples.')
tf.app.flags.DEFINE_boolean('critic_update_dis_vars', False, 'Whether the critic updates the discriminator variables.')
tf.app.flags.DEFINE_string('gen_training_strategy', 'reinforce', "Method for training the Generator. One of ['cross_entropy', 'reinforce']")
tf.app.flags.DEFINE_string('generator_optimizer', 'adam', "Type of Generator optimizer.  One of ['sgd', 'adam']")
tf.app.flags.DEFINE_float('grad_clipping', 10.0, 'Norm for gradient clipping.')
tf.app.flags.DEFINE_float('advantage_clipping', 5.0, 'Clipping for advantages.')
tf.app.flags.DEFINE_string('baseline_method', None, "Approach for baseline.  One of ['critic', 'dis_batch', 'ema', None]")
tf.app.flags.DEFINE_float('perplexity_threshold', 15000, 'Limit for perplexity before terminating job.')
tf.app.flags.DEFINE_float('zoneout_drop_prob', 0.1, 'Probability for dropping parameter for zoneout.')
tf.app.flags.DEFINE_float('keep_prob', 0.5, 'Probability for keeping parameter for dropout.')
tf.app.flags.DEFINE_integer('print_every', 250, 'Frequency to print and log the outputs of the model.')
tf.app.flags.DEFINE_integer('max_num_to_print', 5, 'Number of samples to log/print.')
tf.app.flags.DEFINE_boolean('print_verbose', False, 'Whether to print in full.')
tf.app.flags.DEFINE_integer('summaries_every', 100, 'Frequency to compute summaries.')
tf.app.flags.DEFINE_boolean('eval_language_model', False, 'Whether to evaluate on all words as in language modeling.')
tf.app.flags.DEFINE_float('eval_interval_secs', 60, 'Delay for evaluating model.')
tf.app.flags.DEFINE_integer('n_gram_eval', 4, 'The degree of the n-grams to use for evaluation.')
tf.app.flags.DEFINE_integer('epoch_size_override', None, 'If an integer, this dictates the size of the epochs and will potentially not iterate over all the data.')
tf.app.flags.DEFINE_integer('eval_epoch_size_override', None, 'Number of evaluation steps.')
tf.app.flags.DEFINE_string('base_directory', '/tmp/maskGAN_v0.00', 'Base directory for the logging, events and graph.')
tf.app.flags.DEFINE_string('data_set', 'ptb', "Data set to operate on.  One of['ptb', 'imdb']")
tf.app.flags.DEFINE_string('data_dir', '/tmp/data/ptb', 'Directory for the training data.')
tf.app.flags.DEFINE_string('language_model_ckpt_dir', None, 'Directory storing checkpoints to initialize the model.  Pretrained modelsare stored at /tmp/maskGAN/pretrained/')
tf.app.flags.DEFINE_string('language_model_ckpt_dir_reversed', None, 'Directory storing checkpoints of reversed models to initialize the model.Pretrained models stored atare stored at  /tmp/PTB/pretrained_reversed')
tf.app.flags.DEFINE_string('maskgan_ckpt', None, 'Override which checkpoint file to use to restore the model.  A pretrained seq2seq_zaremba model is stored at /tmp/maskGAN/pretrain/seq2seq_zaremba/train/model.ckpt-64912')
tf.app.flags.DEFINE_boolean('wasserstein_objective', False, '(DEPRECATED) Whether to use the WGAN training.')
tf.app.flags.DEFINE_integer('num_rollouts', 1, 'The number of rolled out predictions to make.')
tf.app.flags.DEFINE_float('c_lower', -0.01, 'Lower bound for weights.')
tf.app.flags.DEFINE_float('c_upper', 0.01, 'Upper bound for weights.')
FLAGS = tf.app.flags.FLAGS

def create_hparams():
    if False:
        for i in range(10):
            print('nop')
    'Create the hparams object for generic training hyperparameters.'
    hparams = tf.contrib.training.HParams(gen_num_layers=2, dis_num_layers=2, gen_rnn_size=740, dis_rnn_size=740, gen_learning_rate=0.0005, dis_learning_rate=0.005, critic_learning_rate=0.005, dis_train_iterations=1, gen_learning_rate_decay=1.0, gen_full_learning_rate_steps=10000000.0, baseline_decay=0.999999, rl_discount_rate=0.9, gen_vd_keep_prob=0.5, dis_vd_keep_prob=0.5, dis_pretrain_learning_rate=0.005, dis_num_filters=128, dis_hidden_dim=128, gen_nas_keep_prob_0=0.85, gen_nas_keep_prob_1=0.55, dis_nas_keep_prob_0=0.85, dis_nas_keep_prob_1=0.55)
    if FLAGS.hparams:
        hparams = hparams.parse(FLAGS.hparams)
    return hparams

def create_MaskGAN(hparams, is_training):
    if False:
        for i in range(10):
            print('nop')
    'Create the MaskGAN model.\n\n  Args:\n    hparams:  Hyperparameters for the MaskGAN.\n    is_training:  Boolean indicating operational mode (train/inference).\n      evaluated with a teacher forcing regime.\n\n  Return:\n    model:  Namedtuple for specifying the MaskGAN.\n  '
    global_step = tf.Variable(0, name='global_step', trainable=False)
    new_learning_rate = tf.placeholder(tf.float32, [], name='new_learning_rate')
    learning_rate = tf.Variable(0.0, name='learning_rate', trainable=False)
    learning_rate_update = tf.assign(learning_rate, new_learning_rate)
    new_rate = tf.placeholder(tf.float32, [], name='new_rate')
    percent_real_var = tf.Variable(0.0, trainable=False)
    percent_real_update = tf.assign(percent_real_var, new_rate)
    inputs = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, FLAGS.sequence_length])
    targets = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, FLAGS.sequence_length])
    present = tf.placeholder(tf.bool, shape=[FLAGS.batch_size, FLAGS.sequence_length])
    real_sequence = targets
    (fake_sequence, fake_logits, fake_log_probs, fake_gen_initial_state, fake_gen_final_state, _) = model_construction.create_generator(hparams, inputs, targets, present, is_training=is_training, is_validating=False)
    (_, eval_logits, _, eval_initial_state, eval_final_state, _) = model_construction.create_generator(hparams, inputs, targets, present, is_training=False, is_validating=True, reuse=True)
    fake_predictions = model_construction.create_discriminator(hparams, fake_sequence, is_training=is_training, inputs=inputs, present=present)
    real_predictions = model_construction.create_discriminator(hparams, real_sequence, is_training=is_training, reuse=True, inputs=inputs, present=present)
    if FLAGS.baseline_method == 'critic':
        est_state_values = model_construction.create_critic(hparams, fake_sequence, is_training=is_training)
    else:
        est_state_values = None
    [dis_loss, dis_loss_fake, dis_loss_real] = model_losses.create_dis_loss(fake_predictions, real_predictions, present)
    avg_log_perplexity = model_losses.calculate_log_perplexity(eval_logits, targets, present)
    fake_cross_entropy_losses = model_losses.create_masked_cross_entropy_loss(targets, present, fake_logits)
    [fake_RL_loss, fake_log_probs, fake_rewards, fake_advantages, fake_baselines, fake_averages_op, critic_loss, cumulative_rewards] = model_losses.calculate_reinforce_objective(hparams, fake_log_probs, fake_predictions, present, est_state_values)
    if FLAGS.gen_pretrain_steps:
        raise NotImplementedError
    else:
        gen_pretrain_op = None
    if FLAGS.dis_pretrain_steps:
        dis_pretrain_op = model_optimization.create_dis_pretrain_op(hparams, dis_loss, global_step)
    else:
        dis_pretrain_op = None
    if FLAGS.gen_training_strategy == 'cross_entropy':
        gen_loss = tf.reduce_mean(fake_cross_entropy_losses)
        [gen_train_op, gen_grads, gen_vars] = model_optimization.create_gen_train_op(hparams, learning_rate, gen_loss, global_step, mode='MINIMIZE')
    elif FLAGS.gen_training_strategy == 'reinforce':
        gen_loss = fake_RL_loss
        [gen_train_op, gen_grads, gen_vars] = model_optimization.create_reinforce_gen_train_op(hparams, learning_rate, gen_loss, fake_averages_op, global_step)
    else:
        raise NotImplementedError
    (dis_train_op, dis_grads, dis_vars) = model_optimization.create_dis_train_op(hparams, dis_loss, global_step)
    if critic_loss is not None:
        [critic_train_op, _, _] = model_optimization.create_critic_train_op(hparams, critic_loss, global_step)
        dis_train_op = tf.group(dis_train_op, critic_train_op)
    with tf.name_scope('general'):
        tf.summary.scalar('percent_real', percent_real_var)
        tf.summary.scalar('learning_rate', learning_rate)
    with tf.name_scope('generator_objectives'):
        tf.summary.scalar('gen_objective', tf.reduce_mean(gen_loss))
        tf.summary.scalar('gen_loss_cross_entropy', tf.reduce_mean(fake_cross_entropy_losses))
    with tf.name_scope('REINFORCE'):
        with tf.name_scope('objective'):
            tf.summary.scalar('fake_RL_loss', tf.reduce_mean(fake_RL_loss))
        with tf.name_scope('rewards'):
            helper.variable_summaries(cumulative_rewards, 'rewards')
        with tf.name_scope('advantages'):
            helper.variable_summaries(fake_advantages, 'advantages')
        with tf.name_scope('baselines'):
            helper.variable_summaries(fake_baselines, 'baselines')
        with tf.name_scope('log_probs'):
            helper.variable_summaries(fake_log_probs, 'log_probs')
    with tf.name_scope('discriminator_losses'):
        tf.summary.scalar('dis_loss', dis_loss)
        tf.summary.scalar('dis_loss_fake_sequence', dis_loss_fake)
        tf.summary.scalar('dis_loss_prob_fake_sequence', tf.exp(-dis_loss_fake))
        tf.summary.scalar('dis_loss_real_sequence', dis_loss_real)
        tf.summary.scalar('dis_loss_prob_real_sequence', tf.exp(-dis_loss_real))
    if critic_loss is not None:
        with tf.name_scope('critic_losses'):
            tf.summary.scalar('critic_loss', critic_loss)
    with tf.name_scope('logits'):
        helper.variable_summaries(fake_logits, 'fake_logits')
    for (v, g) in zip(gen_vars, gen_grads):
        helper.variable_summaries(v, v.op.name)
        helper.variable_summaries(g, 'grad/' + v.op.name)
    for (v, g) in zip(dis_vars, dis_grads):
        helper.variable_summaries(v, v.op.name)
        helper.variable_summaries(g, 'grad/' + v.op.name)
    merge_summaries_op = tf.summary.merge_all()
    text_summary_placeholder = tf.placeholder(tf.string)
    text_summary_op = tf.summary.text('Samples', text_summary_placeholder)
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=5)
    Model = collections.namedtuple('Model', ['inputs', 'targets', 'present', 'percent_real_update', 'new_rate', 'fake_sequence', 'fake_logits', 'fake_rewards', 'fake_baselines', 'fake_advantages', 'fake_log_probs', 'fake_predictions', 'real_predictions', 'fake_cross_entropy_losses', 'fake_gen_initial_state', 'fake_gen_final_state', 'eval_initial_state', 'eval_final_state', 'avg_log_perplexity', 'dis_loss', 'gen_loss', 'critic_loss', 'cumulative_rewards', 'dis_train_op', 'gen_train_op', 'gen_pretrain_op', 'dis_pretrain_op', 'merge_summaries_op', 'global_step', 'new_learning_rate', 'learning_rate_update', 'saver', 'text_summary_op', 'text_summary_placeholder'])
    model = Model(inputs, targets, present, percent_real_update, new_rate, fake_sequence, fake_logits, fake_rewards, fake_baselines, fake_advantages, fake_log_probs, fake_predictions, real_predictions, fake_cross_entropy_losses, fake_gen_initial_state, fake_gen_final_state, eval_initial_state, eval_final_state, avg_log_perplexity, dis_loss, gen_loss, critic_loss, cumulative_rewards, dis_train_op, gen_train_op, gen_pretrain_op, dis_pretrain_op, merge_summaries_op, global_step, new_learning_rate, learning_rate_update, saver, text_summary_op, text_summary_placeholder)
    return model

def compute_geometric_average(percent_captured):
    if False:
        while True:
            i = 10
    'Compute the geometric average of the n-gram metrics.'
    res = 1.0
    for (_, n_gram_percent) in percent_captured.iteritems():
        res *= n_gram_percent
    return np.power(res, 1.0 / float(len(percent_captured)))

def compute_arithmetic_average(percent_captured):
    if False:
        for i in range(10):
            print('nop')
    'Compute the arithmetic average of the n-gram metrics.'
    N = len(percent_captured)
    res = 0.0
    for (_, n_gram_percent) in percent_captured.iteritems():
        res += n_gram_percent
    return res / float(N)

def get_iterator(data):
    if False:
        while True:
            i = 10
    'Return the data iterator.'
    if FLAGS.data_set == 'ptb':
        iterator = ptb_loader.ptb_iterator(data, FLAGS.batch_size, FLAGS.sequence_length, FLAGS.epoch_size_override)
    elif FLAGS.data_set == 'imdb':
        iterator = imdb_loader.imdb_iterator(data, FLAGS.batch_size, FLAGS.sequence_length)
    return iterator

def train_model(hparams, data, log_dir, log, id_to_word, data_ngram_counts):
    if False:
        return 10
    'Train model.\n\n  Args:\n    hparams: Hyperparameters for the MaskGAN.\n    data: Data to evaluate.\n    log_dir: Directory to save checkpoints.\n    log: Readable log for the experiment.\n    id_to_word: Dictionary of indices to words.\n    data_ngram_counts: Dictionary of hashed(n-gram tuples) to counts in the\n      data_set.\n  '
    print('Training model.')
    tf.logging.info('Training model.')
    is_training = True
    log.write('hparams\n')
    log.write(str(hparams))
    log.flush()
    is_chief = FLAGS.task == 0
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
            container_name = ''
            with tf.container(container_name):
                if FLAGS.num_rollouts == 1:
                    model = create_MaskGAN(hparams, is_training)
                elif FLAGS.num_rollouts > 1:
                    model = rollout.create_rollout_MaskGAN(hparams, is_training)
                else:
                    raise ValueError
                print('\nTrainable Variables in Graph:')
                for v in tf.trainable_variables():
                    print(v)
                init_savers = model_utils.retrieve_init_savers(hparams)
                init_fn = partial(model_utils.init_fn, init_savers)
                sv = tf.train.Supervisor(logdir=log_dir, is_chief=is_chief, saver=model.saver, global_step=model.global_step, save_model_secs=60, recovery_wait_secs=30, summary_op=None, init_fn=init_fn)
                with sv.managed_session(FLAGS.master) as sess:
                    if FLAGS.gen_pretrain_steps:
                        pretrain_mask_gan.pretrain_generator(sv, sess, model, data, log, id_to_word, data_ngram_counts, is_chief)
                    if FLAGS.dis_pretrain_steps:
                        pretrain_mask_gan.pretrain_discriminator(sv, sess, model, data, log, id_to_word, data_ngram_counts, is_chief)
                    print_step_division = -1
                    summary_step_division = -1
                    while not sv.ShouldStop():
                        is_present_rate = FLAGS.is_present_rate
                        if FLAGS.is_present_rate_decay is not None:
                            is_present_rate *= 1.0 - FLAGS.is_present_rate_decay
                        model_utils.assign_percent_real(sess, model.percent_real_update, model.new_rate, is_present_rate)
                        (avg_epoch_gen_loss, avg_epoch_dis_loss) = ([], [])
                        cumulative_costs = 0.0
                        gen_iters = 0
                        [gen_initial_state_eval, fake_gen_initial_state_eval] = sess.run([model.eval_initial_state, model.fake_gen_initial_state])
                        dis_initial_state_eval = fake_gen_initial_state_eval
                        zeros_state = fake_gen_initial_state_eval
                        if FLAGS.ps_tasks == 0:
                            dis_offset = 1
                        else:
                            dis_offset = FLAGS.task * 1000 + 1
                        dis_iterator = get_iterator(data)
                        for i in range(dis_offset):
                            try:
                                (dis_x, dis_y, _) = next(dis_iterator)
                            except StopIteration:
                                dis_iterator = get_iterator(data)
                                dis_initial_state_eval = zeros_state
                                (dis_x, dis_y, _) = next(dis_iterator)
                            p = model_utils.generate_mask()
                            train_feed = {model.inputs: dis_x, model.targets: dis_y, model.present: p}
                            if FLAGS.data_set == 'ptb':
                                for (i, (c, h)) in enumerate(model.fake_gen_initial_state):
                                    train_feed[c] = dis_initial_state_eval[i].c
                                    train_feed[h] = dis_initial_state_eval[i].h
                                [dis_initial_state_eval] = sess.run([model.fake_gen_final_state], train_feed)
                        iterator = get_iterator(data)
                        gen_initial_state_eval = zeros_state
                        if FLAGS.ps_tasks > 0:
                            gen_offset = FLAGS.task * 1000 + 1
                            for i in range(gen_offset):
                                try:
                                    next(iterator)
                                except StopIteration:
                                    dis_iterator = get_iterator(data)
                                    dis_initial_state_eval = zeros_state
                                    next(dis_iterator)
                        for (x, y, _) in iterator:
                            for _ in xrange(hparams.dis_train_iterations):
                                try:
                                    (dis_x, dis_y, _) = next(dis_iterator)
                                except StopIteration:
                                    dis_iterator = get_iterator(data)
                                    dis_initial_state_eval = zeros_state
                                    (dis_x, dis_y, _) = next(dis_iterator)
                                    if FLAGS.data_set == 'ptb':
                                        [dis_initial_state_eval] = sess.run([model.fake_gen_initial_state])
                                p = model_utils.generate_mask()
                                train_feed = {model.inputs: dis_x, model.targets: dis_y, model.present: p}
                                if FLAGS.data_set == 'ptb':
                                    for (i, (c, h)) in enumerate(model.fake_gen_initial_state):
                                        train_feed[c] = dis_initial_state_eval[i].c
                                        train_feed[h] = dis_initial_state_eval[i].h
                                (_, dis_loss_eval, step) = sess.run([model.dis_train_op, model.dis_loss, model.global_step], feed_dict=train_feed)
                                [dis_initial_state_eval] = sess.run([model.fake_gen_final_state], train_feed)
                            p = model_utils.generate_mask()
                            train_feed = {model.inputs: x, model.targets: y, model.present: p}
                            if FLAGS.data_set == 'ptb':
                                tf.logging.info('Generator is stateful.')
                                print('Generator is stateful.')
                                for (i, (c, h)) in enumerate(model.eval_initial_state):
                                    train_feed[c] = gen_initial_state_eval[i].c
                                    train_feed[h] = gen_initial_state_eval[i].h
                                for (i, (c, h)) in enumerate(model.fake_gen_initial_state):
                                    train_feed[c] = fake_gen_initial_state_eval[i].c
                                    train_feed[h] = fake_gen_initial_state_eval[i].h
                            lr_decay = hparams.gen_learning_rate_decay ** max(step + 1 - hparams.gen_full_learning_rate_steps, 0.0)
                            gen_learning_rate = hparams.gen_learning_rate * lr_decay
                            model_utils.assign_learning_rate(sess, model.learning_rate_update, model.new_learning_rate, gen_learning_rate)
                            [_, gen_loss_eval, gen_log_perplexity_eval, step] = sess.run([model.gen_train_op, model.gen_loss, model.avg_log_perplexity, model.global_step], feed_dict=train_feed)
                            cumulative_costs += gen_log_perplexity_eval
                            gen_iters += 1
                            [gen_initial_state_eval, fake_gen_initial_state_eval] = sess.run([model.eval_final_state, model.fake_gen_final_state], train_feed)
                            avg_epoch_dis_loss.append(dis_loss_eval)
                            avg_epoch_gen_loss.append(gen_loss_eval)
                            perplexity = np.exp(cumulative_costs / gen_iters)
                            if is_chief and step / FLAGS.summaries_every > summary_step_division:
                                summary_step_division = step / FLAGS.summaries_every
                                if not np.isfinite(perplexity) or perplexity >= FLAGS.perplexity_threshold:
                                    print('Training raising FloatingPoinError.')
                                    raise FloatingPointError('Training infinite perplexity: %.3f' % perplexity)
                                summary_str = sess.run(model.merge_summaries_op, feed_dict=train_feed)
                                sv.SummaryComputed(sess, summary_str)
                                avg_percent_captured = {'2': 0.0, '3': 0.0, '4': 0.0}
                                for (n, data_ngram_count) in data_ngram_counts.iteritems():
                                    batch_percent_captured = evaluation_utils.sequence_ngram_evaluation(sess, model.fake_sequence, log, train_feed, data_ngram_count, int(n))
                                    summary_percent_str = tf.Summary(value=[tf.Summary.Value(tag='general/%s-grams_percent_correct' % n, simple_value=batch_percent_captured)])
                                    sv.SummaryComputed(sess, summary_percent_str, global_step=step)
                                geometric_avg = compute_geometric_average(avg_percent_captured)
                                summary_geometric_avg_str = tf.Summary(value=[tf.Summary.Value(tag='general/geometric_avg', simple_value=geometric_avg)])
                                sv.SummaryComputed(sess, summary_geometric_avg_str, global_step=step)
                                arithmetic_avg = compute_arithmetic_average(avg_percent_captured)
                                summary_arithmetic_avg_str = tf.Summary(value=[tf.Summary.Value(tag='general/arithmetic_avg', simple_value=arithmetic_avg)])
                                sv.SummaryComputed(sess, summary_arithmetic_avg_str, global_step=step)
                                summary_perplexity_str = tf.Summary(value=[tf.Summary.Value(tag='general/perplexity', simple_value=perplexity)])
                                sv.SummaryComputed(sess, summary_perplexity_str, global_step=step)
                            if is_chief and step / FLAGS.print_every > print_step_division:
                                print_step_division = step / FLAGS.print_every
                                print('global_step: %d' % step)
                                print(' perplexity: %.3f' % perplexity)
                                print(' gen_learning_rate: %.6f' % gen_learning_rate)
                                log.write('global_step: %d\n' % step)
                                log.write(' perplexity: %.3f\n' % perplexity)
                                log.write(' gen_learning_rate: %.6f' % gen_learning_rate)
                                avg_percent_captured = {'2': 0.0, '3': 0.0, '4': 0.0}
                                for (n, data_ngram_count) in data_ngram_counts.iteritems():
                                    batch_percent_captured = evaluation_utils.sequence_ngram_evaluation(sess, model.fake_sequence, log, train_feed, data_ngram_count, int(n))
                                    avg_percent_captured[n] = batch_percent_captured
                                    print(' percent of %s-grams captured: %.3f.' % (n, batch_percent_captured))
                                    log.write(' percent of %s-grams captured: %.3f.\n' % (n, batch_percent_captured))
                                geometric_avg = compute_geometric_average(avg_percent_captured)
                                print(' geometric_avg: %.3f.' % geometric_avg)
                                log.write(' geometric_avg: %.3f.' % geometric_avg)
                                arithmetic_avg = compute_arithmetic_average(avg_percent_captured)
                                print(' arithmetic_avg: %.3f.' % arithmetic_avg)
                                log.write(' arithmetic_avg: %.3f.' % arithmetic_avg)
                                evaluation_utils.print_and_log_losses(log, step, is_present_rate, avg_epoch_dis_loss, avg_epoch_gen_loss)
                                if FLAGS.gen_training_strategy == 'reinforce':
                                    evaluation_utils.generate_RL_logs(sess, model, log, id_to_word, train_feed)
                                else:
                                    evaluation_utils.generate_logs(sess, model, log, id_to_word, train_feed)
                                log.flush()
    log.close()

def evaluate_once(data, sv, model, sess, train_dir, log, id_to_word, data_ngram_counts, eval_saver):
    if False:
        while True:
            i = 10
    'Evaluate model for a number of steps.\n\n  Args:\n    data:  Dataset.\n    sv: Supervisor.\n    model: The GAN model we have just built.\n    sess: A session to use.\n    train_dir: Path to a directory containing checkpoints.\n    log: Evaluation log for evaluation.\n    id_to_word: Dictionary of indices to words.\n    data_ngram_counts: Dictionary of hashed(n-gram tuples) to counts in the\n      data_set.\n    eval_saver:  Evaluation saver.r.\n  '
    tf.logging.info('Evaluate Once.')
    model_save_path = tf.latest_checkpoint(train_dir)
    if not model_save_path:
        tf.logging.warning('No checkpoint yet in: %s', train_dir)
        return
    tf.logging.info('Starting eval of: %s' % model_save_path)
    tf.logging.info('Only restoring trainable variables.')
    eval_saver.restore(sess, model_save_path)
    (avg_epoch_gen_loss, avg_epoch_dis_loss) = ([], [])
    cumulative_costs = 0.0
    avg_percent_captured = {'2': 0.0, '3': 0.0, '4': 0.0}
    np.random.seed(0)
    gen_iters = 0
    [gen_initial_state_eval, fake_gen_initial_state_eval] = sess.run([model.eval_initial_state, model.fake_gen_initial_state])
    if FLAGS.eval_language_model:
        is_present_rate = 0.0
        tf.logging.info('Overriding is_present_rate=0. for evaluation.')
        print('Overriding is_present_rate=0. for evaluation.')
    iterator = get_iterator(data)
    for (x, y, _) in iterator:
        if FLAGS.eval_language_model:
            is_present_rate = 0.0
        else:
            is_present_rate = FLAGS.is_present_rate
            tf.logging.info('Evaluating on is_present_rate=%.3f.' % is_present_rate)
        model_utils.assign_percent_real(sess, model.percent_real_update, model.new_rate, is_present_rate)
        p = model_utils.generate_mask()
        eval_feed = {model.inputs: x, model.targets: y, model.present: p}
        if FLAGS.data_set == 'ptb':
            for (i, (c, h)) in enumerate(model.eval_initial_state):
                eval_feed[c] = gen_initial_state_eval[i].c
                eval_feed[h] = gen_initial_state_eval[i].h
            for (i, (c, h)) in enumerate(model.fake_gen_initial_state):
                eval_feed[c] = fake_gen_initial_state_eval[i].c
                eval_feed[h] = fake_gen_initial_state_eval[i].h
        [gen_log_perplexity_eval, dis_loss_eval, gen_loss_eval, gen_initial_state_eval, fake_gen_initial_state_eval, step] = sess.run([model.avg_log_perplexity, model.dis_loss, model.gen_loss, model.eval_final_state, model.fake_gen_final_state, model.global_step], feed_dict=eval_feed)
        for (n, data_ngram_count) in data_ngram_counts.iteritems():
            batch_percent_captured = evaluation_utils.sequence_ngram_evaluation(sess, model.fake_sequence, log, eval_feed, data_ngram_count, int(n))
            avg_percent_captured[n] += batch_percent_captured
        cumulative_costs += gen_log_perplexity_eval
        avg_epoch_dis_loss.append(dis_loss_eval)
        avg_epoch_gen_loss.append(gen_loss_eval)
        gen_iters += 1
    perplexity = np.exp(cumulative_costs / gen_iters)
    for (n, _) in avg_percent_captured.iteritems():
        avg_percent_captured[n] /= gen_iters
    if not np.isfinite(perplexity) or perplexity >= FLAGS.perplexity_threshold:
        print('Evaluation raising FloatingPointError.')
        raise FloatingPointError('Evaluation infinite perplexity: %.3f' % perplexity)
    evaluation_utils.print_and_log_losses(log, step, is_present_rate, avg_epoch_dis_loss, avg_epoch_gen_loss)
    print(' perplexity: %.3f' % perplexity)
    log.write(' perplexity: %.3f\n' % perplexity)
    for (n, n_gram_percent) in avg_percent_captured.iteritems():
        n = int(n)
        print(' percent of %d-grams captured: %.3f.' % (n, n_gram_percent))
        log.write(' percent of %d-grams captured: %.3f.\n' % (n, n_gram_percent))
    samples = evaluation_utils.generate_logs(sess, model, log, id_to_word, eval_feed)
    summary_str = sess.run(model.merge_summaries_op, feed_dict=eval_feed)
    sv.SummaryComputed(sess, summary_str)
    summary_str = sess.run(model.text_summary_op, {model.text_summary_placeholder: '\n\n'.join(samples)})
    sv.SummaryComputed(sess, summary_str, global_step=step)
    for (n, n_gram_percent) in avg_percent_captured.iteritems():
        n = int(n)
        summary_percent_str = tf.Summary(value=[tf.Summary.Value(tag='general/%d-grams_percent_correct' % n, simple_value=n_gram_percent)])
        sv.SummaryComputed(sess, summary_percent_str, global_step=step)
    geometric_avg = compute_geometric_average(avg_percent_captured)
    summary_geometric_avg_str = tf.Summary(value=[tf.Summary.Value(tag='general/geometric_avg', simple_value=geometric_avg)])
    sv.SummaryComputed(sess, summary_geometric_avg_str, global_step=step)
    arithmetic_avg = compute_arithmetic_average(avg_percent_captured)
    summary_arithmetic_avg_str = tf.Summary(value=[tf.Summary.Value(tag='general/arithmetic_avg', simple_value=arithmetic_avg)])
    sv.SummaryComputed(sess, summary_arithmetic_avg_str, global_step=step)
    summary_perplexity_str = tf.Summary(value=[tf.Summary.Value(tag='general/perplexity', simple_value=perplexity)])
    sv.SummaryComputed(sess, summary_perplexity_str, global_step=step)

def evaluate_model(hparams, data, train_dir, log, id_to_word, data_ngram_counts):
    if False:
        for i in range(10):
            print('nop')
    'Evaluate MaskGAN model.\n\n  Args:\n    hparams:  Hyperparameters for the MaskGAN.\n    data: Data to evaluate.\n    train_dir: Path to a directory containing checkpoints.\n    id_to_word: Dictionary of indices to words.\n    data_ngram_counts: Dictionary of hashed(n-gram tuples) to counts in the\n      data_set.\n  '
    tf.logging.error('Evaluate model.')
    is_training = False
    if FLAGS.mode == MODE_VALIDATION:
        logdir = FLAGS.base_directory + '/validation'
    elif FLAGS.mode == MODE_TRAIN_EVAL:
        logdir = FLAGS.base_directory + '/train_eval'
    elif FLAGS.mode == MODE_TEST:
        logdir = FLAGS.base_directory + '/test'
    else:
        raise NotImplementedError
    print(train_dir)
    print(tf.train.latest_checkpoint(train_dir))
    while not tf.train.latest_checkpoint(train_dir):
        tf.logging.error('Waiting for checkpoint...')
        print('Waiting for checkpoint...')
        time.sleep(10)
    with tf.Graph().as_default():
        container_name = ''
        with tf.container(container_name):
            if FLAGS.num_rollouts == 1:
                model = create_MaskGAN(hparams, is_training)
            elif FLAGS.num_rollouts > 1:
                model = rollout.create_rollout_MaskGAN(hparams, is_training)
            else:
                raise ValueError
            evaluation_variables = tf.trainable_variables()
            evaluation_variables.append(model.global_step)
            eval_saver = tf.train.Saver(var_list=evaluation_variables)
            sv = tf.Supervisor(logdir=logdir)
            sess = sv.PrepareSession(FLAGS.eval_master, start_standard_services=False)
            tf.logging.info('Before sv.Loop.')
            sv.Loop(FLAGS.eval_interval_secs, evaluate_once, (data, sv, model, sess, train_dir, log, id_to_word, data_ngram_counts, eval_saver))
            sv.WaitForStop()
            tf.logging.info('sv.Stop().')
            sv.Stop()

def main(_):
    if False:
        print('Hello World!')
    hparams = create_hparams()
    train_dir = FLAGS.base_directory + '/train'
    if FLAGS.data_set == 'ptb':
        raw_data = ptb_loader.ptb_raw_data(FLAGS.data_dir)
        (train_data, valid_data, test_data, _) = raw_data
        valid_data_flat = valid_data
    elif FLAGS.data_set == 'imdb':
        raw_data = imdb_loader.imdb_raw_data(FLAGS.data_dir)
        (train_data, valid_data) = raw_data
        valid_data_flat = [word for review in valid_data for word in review]
    else:
        raise NotImplementedError
    if FLAGS.mode == MODE_TRAIN or FLAGS.mode == MODE_TRAIN_EVAL:
        data_set = train_data
    elif FLAGS.mode == MODE_VALIDATION:
        data_set = valid_data
    elif FLAGS.mode == MODE_TEST:
        data_set = test_data
    else:
        raise NotImplementedError
    if FLAGS.data_set == 'ptb':
        word_to_id = ptb_loader.build_vocab(os.path.join(FLAGS.data_dir, 'ptb.train.txt'))
    elif FLAGS.data_set == 'imdb':
        word_to_id = imdb_loader.build_vocab(os.path.join(FLAGS.data_dir, 'vocab.txt'))
    id_to_word = {v: k for (k, v) in word_to_id.iteritems()}
    bigram_tuples = n_gram.find_all_ngrams(valid_data_flat, n=2)
    trigram_tuples = n_gram.find_all_ngrams(valid_data_flat, n=3)
    fourgram_tuples = n_gram.find_all_ngrams(valid_data_flat, n=4)
    bigram_counts = n_gram.construct_ngrams_dict(bigram_tuples)
    trigram_counts = n_gram.construct_ngrams_dict(trigram_tuples)
    fourgram_counts = n_gram.construct_ngrams_dict(fourgram_tuples)
    print('Unique %d-grams: %d' % (2, len(bigram_counts)))
    print('Unique %d-grams: %d' % (3, len(trigram_counts)))
    print('Unique %d-grams: %d' % (4, len(fourgram_counts)))
    data_ngram_counts = {'2': bigram_counts, '3': trigram_counts, '4': fourgram_counts}
    FLAGS.vocab_size = len(id_to_word)
    print('Vocab size: %d' % FLAGS.vocab_size)
    tf.gfile.MakeDirs(FLAGS.base_directory)
    if FLAGS.mode == MODE_TRAIN:
        log = tf.gfile.GFile(os.path.join(FLAGS.base_directory, 'train-log.txt'), mode='w')
    elif FLAGS.mode == MODE_VALIDATION:
        log = tf.gfile.GFile(os.path.join(FLAGS.base_directory, 'validation-log.txt'), mode='w')
    elif FLAGS.mode == MODE_TRAIN_EVAL:
        log = tf.gfile.GFile(os.path.join(FLAGS.base_directory, 'train_eval-log.txt'), mode='w')
    else:
        log = tf.gfile.GFile(os.path.join(FLAGS.base_directory, 'test-log.txt'), mode='w')
    if FLAGS.mode == MODE_TRAIN:
        train_model(hparams, data_set, train_dir, log, id_to_word, data_ngram_counts)
    elif FLAGS.mode == MODE_VALIDATION:
        evaluate_model(hparams, data_set, train_dir, log, id_to_word, data_ngram_counts)
    elif FLAGS.mode == MODE_TRAIN_EVAL:
        evaluate_model(hparams, data_set, train_dir, log, id_to_word, data_ngram_counts)
    elif FLAGS.mode == MODE_TEST:
        evaluate_model(hparams, data_set, train_dir, log, id_to_word, data_ngram_counts)
    else:
        raise NotImplementedError
if __name__ == '__main__':
    tf.app.run()