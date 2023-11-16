"""Trains the PixelDA model."""
from functools import partial
import os
import tensorflow as tf
from domain_adaptation.datasets import dataset_factory
from domain_adaptation.pixel_domain_adaptation import pixelda_losses
from domain_adaptation.pixel_domain_adaptation import pixelda_model
from domain_adaptation.pixel_domain_adaptation import pixelda_preprocess
from domain_adaptation.pixel_domain_adaptation import pixelda_utils
from domain_adaptation.pixel_domain_adaptation.hparams import create_hparams
slim = tf.contrib.slim
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')
flags.DEFINE_integer('ps_tasks', 0, 'The number of parameter servers. If the value is 0, then the parameters are handled locally by the worker.')
flags.DEFINE_integer('task', 0, 'The Task ID. This value is used when training with multiple workers to identify each worker.')
flags.DEFINE_string('train_log_dir', '/tmp/pixelda/', 'Directory where to write event logs.')
flags.DEFINE_integer('save_summaries_steps', 500, 'The frequency with which summaries are saved, in seconds.')
flags.DEFINE_integer('save_interval_secs', 300, 'The frequency with which the model is saved, in seconds.')
flags.DEFINE_boolean('summarize_gradients', False, 'Whether to summarize model gradients')
flags.DEFINE_integer('print_loss_steps', 100, 'The frequency with which the losses are printed, in steps.')
flags.DEFINE_string('source_dataset', 'mnist', 'The name of the source dataset. If hparams="arch=dcgan", this flag is ignored.')
flags.DEFINE_string('target_dataset', 'mnist_m', 'The name of the target dataset.')
flags.DEFINE_string('source_split_name', 'train', 'Name of the train split for the source.')
flags.DEFINE_string('target_split_name', 'train', 'Name of the train split for the target.')
flags.DEFINE_string('dataset_dir', '', 'The directory where the datasets can be found.')
flags.DEFINE_integer('num_readers', 4, 'The number of parallel readers that read data from the dataset.')
flags.DEFINE_integer('num_preprocessing_threads', 4, 'The number of threads used to create the batches.')
flags.DEFINE_string('hparams', '', 'Comma separated hyperparameter values')

def _get_vars_and_update_ops(hparams, scope):
    if False:
        return 10
    'Returns the variables and update ops for a particular variable scope.\n\n  Args:\n    hparams: The hyperparameters struct.\n    scope: The variable scope.\n\n  Returns:\n    A tuple consisting of trainable variables and update ops.\n  '
    is_trainable = lambda x: x in tf.trainable_variables()
    var_list = filter(is_trainable, slim.get_model_variables(scope))
    global_step = slim.get_or_create_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
    tf.logging.info('All variables for scope: %s', slim.get_model_variables(scope))
    tf.logging.info('Trainable variables for scope: %s', var_list)
    return (var_list, update_ops)

def _train(discriminator_train_op, generator_train_op, logdir, master='', is_chief=True, scaffold=None, hooks=None, chief_only_hooks=None, save_checkpoint_secs=600, save_summaries_steps=100, hparams=None):
    if False:
        while True:
            i = 10
    "Runs the training loop.\n\n  Args:\n    discriminator_train_op: A `Tensor` that, when executed, will apply the\n      gradients and return the loss value for the discriminator.\n    generator_train_op: A `Tensor` that, when executed, will apply the\n      gradients and return the loss value for the generator.\n    logdir: The directory where the graph and checkpoints are saved.\n    master: The URL of the master.\n    is_chief: Specifies whether or not the training is being run by the primary\n      replica during replica training.\n    scaffold: An tf.train.Scaffold instance.\n    hooks: List of `tf.train.SessionRunHook` callbacks which are run inside the\n      training loop.\n    chief_only_hooks: List of `tf.train.SessionRunHook` instances which are run\n      inside the training loop for the chief trainer only.\n    save_checkpoint_secs: The frequency, in seconds, that a checkpoint is saved\n      using a default checkpoint saver. If `save_checkpoint_secs` is set to\n      `None`, then the default checkpoint saver isn't used.\n    save_summaries_steps: The frequency, in number of global steps, that the\n      summaries are written to disk using a default summary saver. If\n      `save_summaries_steps` is set to `None`, then the default summary saver\n      isn't used.\n    hparams: The hparams struct.\n\n  Returns:\n    the value of the loss function after training.\n\n  Raises:\n    ValueError: if `logdir` is `None` and either `save_checkpoint_secs` or\n    `save_summaries_steps` are `None.\n  "
    global_step = slim.get_or_create_global_step()
    scaffold = scaffold or tf.train.Scaffold()
    hooks = hooks or []
    if is_chief:
        session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold, checkpoint_dir=logdir, master=master)
        if chief_only_hooks:
            hooks.extend(chief_only_hooks)
        hooks.append(tf.train.StepCounterHook(output_dir=logdir))
        if save_summaries_steps:
            if logdir is None:
                raise ValueError('logdir cannot be None when save_summaries_steps is None')
            hooks.append(tf.train.SummarySaverHook(scaffold=scaffold, save_steps=save_summaries_steps, output_dir=logdir))
        if save_checkpoint_secs:
            if logdir is None:
                raise ValueError('logdir cannot be None when save_checkpoint_secs is None')
            hooks.append(tf.train.CheckpointSaverHook(logdir, save_secs=save_checkpoint_secs, scaffold=scaffold))
    else:
        session_creator = tf.train.WorkerSessionCreator(scaffold=scaffold, master=master)
    with tf.train.MonitoredSession(session_creator=session_creator, hooks=hooks) as session:
        loss = None
        while not session.should_stop():
            for _ in range(hparams.discriminator_steps):
                if session.should_stop():
                    return loss
                (loss, np_global_step) = session.run([discriminator_train_op, global_step])
                if np_global_step % FLAGS.print_loss_steps == 0:
                    tf.logging.info('Step %d: Discriminator Loss = %.2f', np_global_step, loss)
            for _ in range(hparams.generator_steps):
                if session.should_stop():
                    return loss
                (loss, np_global_step) = session.run([generator_train_op, global_step])
                if np_global_step % FLAGS.print_loss_steps == 0:
                    tf.logging.info('Step %d: Generator Loss = %.2f', np_global_step, loss)
    return loss

def run_training(run_dir, checkpoint_dir, hparams):
    if False:
        while True:
            i = 10
    'Runs the training loop.\n\n  Args:\n    run_dir: The directory where training specific logs are placed\n    checkpoint_dir: The directory where the checkpoints and log files are\n      stored.\n    hparams: The hyperparameters struct.\n\n  Raises:\n    ValueError: if hparams.arch is not recognized.\n  '
    for path in [run_dir, checkpoint_dir]:
        if not tf.gfile.Exists(path):
            tf.gfile.MakeDirs(path)
    hparams_filename = os.path.join(checkpoint_dir, 'hparams.json')
    with tf.gfile.FastGFile(hparams_filename, 'w') as f:
        f.write(hparams.to_json())
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
            global_step = slim.get_or_create_global_step()
            target_dataset = dataset_factory.get_dataset(FLAGS.target_dataset, split_name='train', dataset_dir=FLAGS.dataset_dir)
            (target_images, _) = dataset_factory.provide_batch(FLAGS.target_dataset, 'train', FLAGS.dataset_dir, FLAGS.num_readers, hparams.batch_size, FLAGS.num_preprocessing_threads)
            num_target_classes = target_dataset.num_classes
            if hparams.arch not in ['dcgan']:
                source_dataset = dataset_factory.get_dataset(FLAGS.source_dataset, split_name='train', dataset_dir=FLAGS.dataset_dir)
                num_source_classes = source_dataset.num_classes
                (source_images, source_labels) = dataset_factory.provide_batch(FLAGS.source_dataset, 'train', FLAGS.dataset_dir, FLAGS.num_readers, hparams.batch_size, FLAGS.num_preprocessing_threads)
                source_labels['class'] = tf.argmax(source_labels['classes'], 1)
                del source_labels['classes']
                if num_source_classes != num_target_classes:
                    raise ValueError('Source and Target datasets must have same number of classes. Are %d and %d' % (num_source_classes, num_target_classes))
            else:
                source_images = None
                source_labels = None
            end_points = pixelda_model.create_model(hparams, target_images, source_images=source_images, source_labels=source_labels, is_training=True, num_classes=num_target_classes)
            (generator_vars, generator_update_ops) = _get_vars_and_update_ops(hparams, 'generator')
            (discriminator_vars, discriminator_update_ops) = _get_vars_and_update_ops(hparams, 'discriminator')
            generator_loss = pixelda_losses.g_step_loss(source_images, source_labels, end_points, hparams, num_classes=num_target_classes)
            discriminator_loss = pixelda_losses.d_step_loss(end_points, source_labels, num_target_classes, hparams)
            learning_rate = hparams.learning_rate
            if hparams.lr_decay_steps:
                learning_rate = tf.train.exponential_decay(learning_rate, slim.get_or_create_global_step(), decay_steps=hparams.lr_decay_steps, decay_rate=hparams.lr_decay_rate, staircase=True)
            tf.summary.scalar('Learning_rate', learning_rate)
            if hparams.discriminator_steps == 0:
                discriminator_train_op = tf.no_op()
            else:
                discriminator_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=hparams.adam_beta1)
                discriminator_train_op = slim.learning.create_train_op(discriminator_loss, discriminator_optimizer, update_ops=discriminator_update_ops, variables_to_train=discriminator_vars, clip_gradient_norm=hparams.clip_gradient_norm, summarize_gradients=FLAGS.summarize_gradients)
            if hparams.generator_steps == 0:
                generator_train_op = tf.no_op()
            else:
                generator_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=hparams.adam_beta1)
                generator_train_op = slim.learning.create_train_op(generator_loss, generator_optimizer, update_ops=generator_update_ops, variables_to_train=generator_vars, clip_gradient_norm=hparams.clip_gradient_norm, summarize_gradients=FLAGS.summarize_gradients)
            pixelda_utils.summarize_model(end_points)
            pixelda_utils.summarize_transferred_grid(end_points['transferred_images'], source_images, name='Transferred')
            if 'source_images_recon' in end_points:
                pixelda_utils.summarize_transferred_grid(end_points['source_images_recon'], source_images, name='Source Reconstruction')
            pixelda_utils.summaries_color_distributions(end_points['transferred_images'], 'Transferred')
            pixelda_utils.summaries_color_distributions(target_images, 'Target')
            if source_images is not None:
                pixelda_utils.summarize_transferred(source_images, end_points['transferred_images'])
                pixelda_utils.summaries_color_distributions(source_images, 'Source')
                pixelda_utils.summaries_color_distributions(tf.abs(source_images - end_points['transferred_images']), 'Abs(Source_minus_Transferred)')
            number_of_steps = None
            if hparams.num_training_examples:
                number_of_steps = hparams.num_training_examples / hparams.batch_size
            hooks = [tf.train.StepCounterHook()]
            chief_only_hooks = [tf.train.CheckpointSaverHook(saver=tf.train.Saver(), checkpoint_dir=run_dir, save_secs=FLAGS.save_interval_secs)]
            if number_of_steps:
                hooks.append(tf.train.StopAtStepHook(last_step=number_of_steps))
            _train(discriminator_train_op, generator_train_op, logdir=run_dir, master=FLAGS.master, is_chief=FLAGS.task == 0, hooks=hooks, chief_only_hooks=chief_only_hooks, save_checkpoint_secs=None, save_summaries_steps=FLAGS.save_summaries_steps, hparams=hparams)

def main(_):
    if False:
        i = 10
        return i + 15
    tf.logging.set_verbosity(tf.logging.INFO)
    hparams = create_hparams(FLAGS.hparams)
    run_training(run_dir=FLAGS.train_log_dir, checkpoint_dir=FLAGS.train_log_dir, hparams=hparams)
if __name__ == '__main__':
    tf.app.run()