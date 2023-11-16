from __future__ import print_function
import tensorflow as tf
import numpy as np
from auxiliary import progress_bar
import os
import sys

def train(**keywords):
    if False:
        i = 10
        return i + 15
    '\n    This function run the session whether in training or evaluation mode.\n    NOTE: **keywords is defined in order to make the code easily changable.\n    WARNING: All the arguments for the **keywords must be defined when calling this function.\n    **keywords:\n    :param sess: The default session.\n    :param saver: The saver operator to save and load the model weights.\n    :param tensors: The tensors dictionary defined by the graph.\n    :param data: The data structure.\n    :param train_dir: The training dir which is a reference for saving the logs and model checkpoints.\n    :param finetuning: If fine tuning should be done or random initialization is needed.\n    :param num_epochs: Number of epochs for training.\n    :param online_test: If the testing is done while training.\n    :param checkpoint_dir: The directory of the checkpoints.\n    :param batch_size: The training batch size.\n\n    :return:\n             Run the session.\n    '
    checkpoint_prefix = 'model'
    train_summary_dir = os.path.join(keywords['train_dir'], 'summaries', 'train')
    train_summary_writer = tf.summary.FileWriter(train_summary_dir)
    train_summary_writer.add_graph(keywords['sess'].graph)
    test_summary_dir = os.path.join(keywords['train_dir'], 'summaries', 'test')
    test_summary_writer = tf.summary.FileWriter(test_summary_dir)
    test_summary_writer.add_graph(keywords['sess'].graph)
    if keywords['finetuning']:
        keywords['saver'].restore(keywords['sess'], os.path.join(keywords['checkpoint_dir'], checkpoint_prefix))
        print('Model restored for fine-tuning...')
    for epoch in range(keywords['num_epochs']):
        total_batch_training = int(keywords['data'].train.images.shape[0] / keywords['batch_size'])
        for batch_num in range(total_batch_training):
            start_idx = batch_num * keywords['batch_size']
            end_idx = (batch_num + 1) * keywords['batch_size']
            (train_batch_data, train_batch_label) = (keywords['data'].train.images[start_idx:end_idx], keywords['data'].train.labels[start_idx:end_idx])
            (batch_loss, _, train_summaries, training_step) = keywords['sess'].run([keywords['tensors']['cost'], keywords['tensors']['train_op'], keywords['tensors']['summary_train_op'], keywords['tensors']['global_step']], feed_dict={keywords['tensors']['image_place']: train_batch_data, keywords['tensors']['label_place']: train_batch_label, keywords['tensors']['dropout_param']: 0.5})
            train_summary_writer.add_summary(train_summaries, global_step=training_step)
            progress = float(batch_num + 1) / total_batch_training
            progress_bar.print_progress(progress, epoch_num=epoch + 1, loss=batch_loss)
        summary_epoch_train_op = keywords['tensors']['summary_epoch_train_op']
        train_epoch_summaries = keywords['sess'].run(summary_epoch_train_op, feed_dict={keywords['tensors']['image_place']: train_batch_data, keywords['tensors']['label_place']: train_batch_label, keywords['tensors']['dropout_param']: 1.0})
        train_summary_writer.add_summary(train_epoch_summaries, global_step=training_step)
        if keywords['online_test']:
            (test_accuracy_epoch, test_summaries) = keywords['sess'].run([keywords['tensors']['accuracy'], keywords['tensors']['summary_test_op']], feed_dict={keywords['tensors']['image_place']: keywords['data'].test.images, keywords['tensors']['label_place']: keywords['data'].test.labels, keywords['tensors']['dropout_param']: 1.0})
            print('Epoch ' + str(epoch + 1) + ', Testing Accuracy= ' + '{:.5f}'.format(test_accuracy_epoch))
            current_step = tf.train.global_step(keywords['sess'], keywords['tensors']['global_step'])
            test_summary_writer.add_summary(test_summaries, global_step=current_step)
    if not os.path.exists(keywords['checkpoint_dir']):
        os.makedirs(keywords['checkpoint_dir'])
    save_path = keywords['saver'].save(keywords['sess'], os.path.join(keywords['checkpoint_dir'], checkpoint_prefix))
    print('Model saved in file: %s' % save_path)

def evaluation(**keywords):
    if False:
        while True:
            i = 10
    checkpoint_prefix = 'model'
    saver = keywords['saver']
    sess = keywords['sess']
    checkpoint_dir = keywords['checkpoint_dir']
    data = keywords['data']
    accuracy_tensor = keywords['tensors']['accuracy']
    image_place = keywords['tensors']['image_place']
    label_place = keywords['tensors']['label_place']
    dropout_param = keywords['tensors']['dropout_param']
    saver.restore(sess, os.path.join(checkpoint_dir, checkpoint_prefix))
    print('Model restored...')
    test_set = data.test.images
    test_label = data.test.labels
    test_accuracy = 100 * keywords['sess'].run(accuracy_tensor, feed_dict={image_place: test_set, label_place: test_label, dropout_param: 1.0})
    print('Final Test Accuracy is %% %.2f' % test_accuracy)