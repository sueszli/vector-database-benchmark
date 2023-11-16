"""Trains the integrated LexNET classifier."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import lexnet_common
import lexnet_model
import path_model
from sklearn import metrics
import tensorflow as tf
tf.flags.DEFINE_string('dataset_dir', 'datasets', 'Dataset base directory')
tf.flags.DEFINE_string('dataset', 'tratz/fine_grained', 'Subdirectory containing the corpus directories: subdirectory of dataset_dir')
tf.flags.DEFINE_string('corpus', 'wiki/random', 'Subdirectory containing the corpus and split: subdirectory of dataset_dir/dataset')
tf.flags.DEFINE_string('embeddings_base_path', 'embeddings', 'Embeddings base directory')
tf.flags.DEFINE_string('logdir', 'logdir', 'Directory of model output files')
tf.flags.DEFINE_string('hparams', '', 'Hyper-parameters')
tf.flags.DEFINE_string('input', 'integrated', 'The model(dist/dist-nc/path/integrated/integrated-nc')
FLAGS = tf.flags.FLAGS

def main(_):
    if False:
        while True:
            i = 10
    hparams = lexnet_model.LexNETModel.default_hparams()
    hparams.corpus = FLAGS.corpus
    hparams.input = FLAGS.input
    hparams.path_embeddings_file = 'path_embeddings/%s/%s' % (FLAGS.dataset, FLAGS.corpus)
    input_dir = hparams.input if hparams.input != 'path' else 'path_classifier'
    classes_filename = os.path.join(FLAGS.dataset_dir, FLAGS.dataset, 'classes.txt')
    with open(classes_filename) as f_in:
        classes = f_in.read().splitlines()
    hparams.num_classes = len(classes)
    print('Model will predict into %d classes' % hparams.num_classes)
    (train_set, val_set, test_set) = (os.path.join(FLAGS.dataset_dir, FLAGS.dataset, FLAGS.corpus, filename + '.tfrecs.gz') for filename in ['train', 'val', 'test'])
    print('Running with hyper-parameters: {}'.format(hparams))
    print('Loading instances...')
    opts = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    train_instances = list(tf.python_io.tf_record_iterator(train_set, opts))
    val_instances = list(tf.python_io.tf_record_iterator(val_set, opts))
    test_instances = list(tf.python_io.tf_record_iterator(test_set, opts))
    print('Loading word embeddings...')
    (relata_embeddings, path_embeddings, nc_embeddings, path_to_index) = (None, None, None, None)
    if hparams.input in ['dist', 'dist-nc', 'integrated', 'integrated-nc']:
        relata_embeddings = lexnet_common.load_word_embeddings(FLAGS.embeddings_base_path, hparams.relata_embeddings_file)
    if hparams.input in ['path', 'integrated', 'integrated-nc']:
        (path_embeddings, path_to_index) = path_model.load_path_embeddings(os.path.join(FLAGS.embeddings_base_path, hparams.path_embeddings_file), hparams.path_dim)
    if hparams.input in ['dist-nc', 'integrated-nc']:
        nc_embeddings = lexnet_common.load_word_embeddings(FLAGS.embeddings_base_path, hparams.nc_embeddings_file)
    with tf.Graph().as_default():
        model = lexnet_model.LexNETModel(hparams, relata_embeddings, path_embeddings, nc_embeddings, path_to_index)
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        if hparams.input in ['path', 'integrated', 'integrated-nc']:
            session.run(tf.tables_initializer())
            session.run(model.initialize_path_op, {model.path_initial_value_t: path_embeddings})
        if hparams.input in ['dist-nc', 'integrated-nc']:
            session.run(model.initialize_nc_op, {model.nc_initial_value_t: nc_embeddings})
        print('Loading labels...')
        train_labels = model.load_labels(session, train_instances)
        val_labels = model.load_labels(session, val_instances)
        test_labels = model.load_labels(session, test_instances)
        save_path = '{logdir}/results/{dataset}/{input}/{corpus}'.format(logdir=FLAGS.logdir, dataset=FLAGS.dataset, corpus=model.hparams.corpus, input=input_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print('Training the model...')
        model.fit(session, train_instances, epoch_completed, val_instances, val_labels, save_path)
        print('Best performance on the validation set: F1=%.3f' % epoch_completed.best_f1)
        lexnet_common.full_evaluation(model, session, train_instances, train_labels, 'Train', classes)
        lexnet_common.full_evaluation(model, session, val_instances, val_labels, 'Validation', classes)
        test_predictions = lexnet_common.full_evaluation(model, session, test_instances, test_labels, 'Test', classes)
        predictions_file = os.path.join(save_path, 'test_predictions.tsv')
        print('Saving test predictions to %s' % save_path)
        test_pairs = model.load_pairs(session, test_instances)
        lexnet_common.write_predictions(test_pairs, test_labels, test_predictions, classes, predictions_file)

def epoch_completed(model, session, epoch, epoch_loss, val_instances, val_labels, save_path):
    if False:
        print('Hello World!')
    'Runs every time an epoch completes.\n\n  Print the performance on the validation set, and update the saved model if\n  its performance is better on the previous ones. If the performance dropped,\n  tell the training to stop.\n\n  Args:\n    model: The currently trained path-based model.\n    session: The current TensorFlow session.\n    epoch: The epoch number.\n    epoch_loss: The current epoch loss.\n    val_instances: The validation set instances (evaluation between epochs).\n    val_labels: The validation set labels (for evaluation between epochs).\n    save_path: Where to save the model.\n\n  Returns:\n    whether the training should stop.\n  '
    stop_training = False
    val_pred = model.predict(session, val_instances)
    (precision, recall, f1, _) = metrics.precision_recall_fscore_support(val_labels, val_pred, average='weighted')
    print('Epoch: %d/%d, Loss: %f, validation set: P: %.3f, R: %.3f, F1: %.3f\n' % (epoch + 1, model.hparams.num_epochs, epoch_loss, precision, recall, f1))
    if f1 < epoch_completed.best_f1 - 0.08:
        stop_training = True
    if f1 > epoch_completed.best_f1:
        saver = tf.train.Saver()
        checkpoint_filename = os.path.join(save_path, 'best.ckpt')
        print('Saving model in: %s' % checkpoint_filename)
        saver.save(session, checkpoint_filename)
        print('Model saved in file: %s' % checkpoint_filename)
        epoch_completed.best_f1 = f1
    return stop_training
epoch_completed.best_f1 = 0
if __name__ == '__main__':
    tf.app.run(main)