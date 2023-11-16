"""Utility functions to build DRAGNN MasterSpecs and schedule model training.

Provides functions to finish a MasterSpec, building required lexicons for it and
adding them as resources, as well as setting features sizes.
"""
import random
import tensorflow as tf
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.python.framework import errors
from tensorflow.python.platform import gfile
from syntaxnet.util import check

def calculate_component_accuracies(eval_res_values):
    if False:
        while True:
            i = 10
    'Transforms the DRAGNN eval_res output to float accuracies of components.'
    return [1.0 * eval_res_values[2 * i + 1] / eval_res_values[2 * i] if eval_res_values[2 * i] > 0 else float('nan') for i in xrange(len(eval_res_values) // 2)]

def write_summary(summary_writer, label, value, step):
    if False:
        for i in range(10):
            print('nop')
    'Write a summary for a certain evaluation.'
    summary = Summary(value=[Summary.Value(tag=label, simple_value=float(value))])
    summary_writer.add_summary(summary, step)
    summary_writer.flush()

def annotate_dataset(sess, annotator, eval_corpus):
    if False:
        print('Hello World!')
    'Annotate eval_corpus given a model.'
    batch_size = min(len(eval_corpus), 1024)
    processed = []
    tf.logging.info('Annotating datset: %d examples', len(eval_corpus))
    for start in range(0, len(eval_corpus), batch_size):
        end = min(start + batch_size, len(eval_corpus))
        serialized_annotations = sess.run(annotator['annotations'], feed_dict={annotator['input_batch']: eval_corpus[start:end]})
        assert len(serialized_annotations) == end - start
        processed.extend(serialized_annotations)
    tf.logging.info('Done. Produced %d annotations', len(processed))
    return processed

def get_summary_writer(tensorboard_dir):
    if False:
        print('Hello World!')
    'Creates a directory for writing summaries and returns a writer.'
    tf.logging.info('TensorBoard directory: %s', tensorboard_dir)
    tf.logging.info('Deleting prior data if exists...')
    try:
        gfile.DeleteRecursively(tensorboard_dir)
    except errors.OpError as err:
        tf.logging.error('Directory did not exist? Error: %s', err)
    tf.logging.info('Deleted! Creating the directory again...')
    gfile.MakeDirs(tensorboard_dir)
    tf.logging.info('Created! Instatiating SummaryWriter...')
    summary_writer = tf.summary.FileWriter(tensorboard_dir)
    return summary_writer

def generate_target_per_step_schedule(pretrain_steps, train_steps):
    if False:
        for i in range(10):
            print('nop')
    'Generates a sampled training schedule.\n\n  Arguments:\n    pretrain_steps: List, number of pre-training steps per each target.\n    train_steps: List, number of sampled training steps per each target.\n\n  Returns:\n    Python list of length sum(pretrain_steps + train_steps), containing\n    target numbers per step.\n  '
    check.Eq(len(pretrain_steps), len(train_steps))
    random.seed(201527)
    tf.logging.info('Determining the training schedule...')
    target_per_step = []
    for target_idx in xrange(len(pretrain_steps)):
        target_per_step += [target_idx] * pretrain_steps[target_idx]
    train_steps = list(train_steps)
    while sum(train_steps) > 0:
        step = random.randint(0, sum(train_steps) - 1)
        cumulative_steps = 0
        for target_idx in xrange(len(train_steps)):
            cumulative_steps += train_steps[target_idx]
            if step < cumulative_steps:
                break
        assert train_steps[target_idx] > 0
        train_steps[target_idx] -= 1
        target_per_step.append(target_idx)
    tf.logging.info('Training schedule defined!')
    return target_per_step

def run_training_step(sess, trainer, train_corpus, batch_size):
    if False:
        i = 10
        return i + 15
    'Runs a single iteration of train_op on a randomly sampled batch.'
    batch = random.sample(train_corpus, batch_size)
    sess.run(trainer['run'], feed_dict={trainer['input_batch']: batch})

def run_training(sess, trainers, annotator, evaluator, pretrain_steps, train_steps, train_corpus, eval_corpus, eval_gold, batch_size, summary_writer, report_every, saver, checkpoint_filename, checkpoint_stats=None):
    if False:
        for i in range(10):
            print('nop')
    "Runs multi-task DRAGNN training on a single corpus.\n\n  Arguments:\n    sess: TF session to use.\n    trainers: List of training ops to use.\n    annotator: Annotation op.\n    evaluator: Function taking two serialized corpora and returning a dict of\n      scalar summaries representing evaluation metrics. The 'eval_metric'\n      summary will be used for early stopping.\n    pretrain_steps: List of the no. of pre-training steps for each train op.\n    train_steps: List of the total no. of steps for each train op.\n    train_corpus: Training corpus to use.\n    eval_corpus: Holdout Corpus for early stoping.\n    eval_gold: Reference of eval_corpus for computing accuracy.\n      eval_corpus and eval_gold are allowed to be the same if eval_corpus\n      already contains gold annotation.\n      Note for segmentation eval_corpus and eval_gold are always different since\n      eval_corpus contains sentences whose tokens are utf8-characters while\n      eval_gold's tokens are gold words.\n    batch_size: How many examples to send to the train op at a time.\n    summary_writer: TF SummaryWriter to use to write summaries.\n    report_every: How often to compute summaries (in steps).\n    saver: TF saver op to save variables.\n    checkpoint_filename: File to save checkpoints to.\n    checkpoint_stats: Stats of checkpoint.\n  "
    if not checkpoint_stats:
        checkpoint_stats = [0] * (len(train_steps) + 1)
    target_per_step = generate_target_per_step_schedule(pretrain_steps, train_steps)
    best_eval_metric = -1.0
    tf.logging.info('Starting training...')
    actual_step = sum(checkpoint_stats[1:])
    for (step, target_idx) in enumerate(target_per_step):
        run_training_step(sess, trainers[target_idx], train_corpus, batch_size)
        checkpoint_stats[target_idx + 1] += 1
        if step % 100 == 0:
            tf.logging.info('training step: %d, actual: %d', step, actual_step + step)
        if step % report_every == 0:
            tf.logging.info('finished step: %d, actual: %d', step, actual_step + step)
            annotated = annotate_dataset(sess, annotator, eval_corpus)
            summaries = evaluator(eval_gold, annotated)
            for (label, metric) in summaries.iteritems():
                write_summary(summary_writer, label, metric, actual_step + step)
            eval_metric = summaries['eval_metric']
            tf.logging.info('Current eval metric: %.2f', eval_metric)
            if best_eval_metric < eval_metric:
                tf.logging.info('Updating best eval to %.2f, saving checkpoint.', eval_metric)
                best_eval_metric = eval_metric
                saver.save(sess, checkpoint_filename)
                with gfile.GFile('%s.stats' % checkpoint_filename, 'w') as f:
                    stats_str = ','.join([str(x) for x in checkpoint_stats])
                    f.write(stats_str)
                    tf.logging.info('Writing stats: %s', stats_str)
    tf.logging.info('Finished training!')