import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))
import numpy as np
import tensorflow as tf
from config import get_config
from model_n2nmn.assembler import Assembler
from model_n2nmn.model import Model
from util.data_reader import DataReader
from util.data_reader import SampleBuilder
from util.misc import prepare_dirs_and_logger
from util.misc import save_config
from util.misc import show_all_variables

def main(_):
    if False:
        i = 10
        return i + 15
    config = prepare_dirs_and_logger(config_raw)
    save_config(config)
    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)
    config.rng = rng
    config.module_names = ['_key_find', '_key_filter', '_val_desc', '<eos>']
    config.gt_layout_tokens = ['_key_find', '_key_filter', '_val_desc', '<eos>']
    assembler = Assembler(config)
    sample_builder = SampleBuilder(config)
    config = sample_builder.config
    data_train = sample_builder.data_all['train']
    data_reader_train = DataReader(config, data_train, assembler, shuffle=True, one_pass=False)
    num_vocab_txt = len(sample_builder.dict_all)
    num_vocab_nmn = len(assembler.module_names)
    num_choices = len(sample_builder.dict_all)
    text_seq_batch = tf.placeholder(tf.int32, [None, None])
    seq_len_batch = tf.placeholder(tf.int32, [None])
    ans_label_batch = tf.placeholder(tf.int32, [None])
    use_gt_layout = tf.constant(True, dtype=tf.bool)
    gt_layout_batch = tf.placeholder(tf.int32, [None, None])
    model = Model(config, sample_builder.kb, text_seq_batch, seq_len_batch, num_vocab_txt=num_vocab_txt, num_vocab_nmn=num_vocab_nmn, EOS_idx=assembler.EOS_idx, num_choices=num_choices, decoder_sampling=True, use_gt_layout=use_gt_layout, gt_layout_batch=gt_layout_batch)
    compiler = model.compiler
    scores = model.scores
    log_seq_prob = model.log_seq_prob
    softmax_loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=ans_label_batch)
    final_loss_per_sample = softmax_loss_per_sample
    avg_sample_loss = tf.reduce_mean(final_loss_per_sample)
    seq_likelihood_loss = tf.reduce_mean(-log_seq_prob)
    total_training_loss = seq_likelihood_loss + avg_sample_loss
    total_loss = total_training_loss + config.weight_decay * model.l2_reg
    solver = tf.train.AdamOptimizer()
    gradients = solver.compute_gradients(total_loss)
    gradients = [(tf.clip_by_norm(g, config.max_grad_norm), v) for (g, v) in gradients]
    solver_op = solver.apply_gradients(gradients)
    with tf.control_dependencies([solver_op]):
        train_step = tf.constant(0)
    log_writer = tf.summary.FileWriter(config.log_dir, tf.get_default_graph())
    loss_ph = tf.placeholder(tf.float32, [])
    entropy_ph = tf.placeholder(tf.float32, [])
    accuracy_ph = tf.placeholder(tf.float32, [])
    summary_train = [tf.summary.scalar('avg_sample_loss', loss_ph), tf.summary.scalar('entropy', entropy_ph), tf.summary.scalar('avg_accuracy', accuracy_ph)]
    log_step_train = tf.summary.merge(summary_train)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    snapshot_saver = tf.train.Saver(max_to_keep=None)
    show_all_variables()
    avg_accuracy = 0
    accuracy_decay = 0.99
    for (n_iter, batch) in enumerate(data_reader_train.batches()):
        if n_iter >= config.max_iter:
            break
        h = sess.partial_run_setup(fetches=[model.predicted_tokens, model.entropy_reg, scores, avg_sample_loss, train_step], feeds=[text_seq_batch, seq_len_batch, gt_layout_batch, compiler.loom_input_tensor, ans_label_batch])
        (tokens, entropy_reg_val) = sess.partial_run(h, fetches=(model.predicted_tokens, model.entropy_reg), feed_dict={text_seq_batch: batch['input_seq_batch'], seq_len_batch: batch['seq_len_batch'], gt_layout_batch: batch['gt_layout_batch']})
        (expr_list, expr_validity_array) = assembler.assemble(tokens)
        assert np.all(expr_validity_array)
        labels = batch['ans_label_batch']
        expr_feed = compiler.build_feed_dict(expr_list)
        expr_feed[ans_label_batch] = labels
        (scores_val, avg_sample_loss_val, _) = sess.partial_run(h, fetches=(scores, avg_sample_loss, train_step), feed_dict=expr_feed)
        predictions = np.argmax(scores_val, axis=1)
        accuracy = np.mean(np.logical_and(expr_validity_array, predictions == labels))
        avg_accuracy += (1 - accuracy_decay) * (accuracy - avg_accuracy)
        if (n_iter + 1) % config.log_interval == 0:
            tf.logging.info('iter = %d\n\tloss = %f, accuracy (cur) = %f, accuracy (avg) = %f, entropy = %f' % (n_iter + 1, avg_sample_loss_val, accuracy, avg_accuracy, -entropy_reg_val))
            summary = sess.run(fetches=log_step_train, feed_dict={loss_ph: avg_sample_loss_val, entropy_ph: -entropy_reg_val, accuracy_ph: avg_accuracy})
            log_writer.add_summary(summary, n_iter + 1)
        if (n_iter + 1) % config.snapshot_interval == 0:
            snapshot_file = os.path.join(config.model_dir, '%08d' % (n_iter + 1))
            snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
            tf.logging.info('Snapshot saved to %s' % snapshot_file)
    tf.logging.info('Run finished.')
if __name__ == '__main__':
    (config_raw, unparsed) = get_config()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)