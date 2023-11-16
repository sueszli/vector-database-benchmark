"""Script which evaluates model on adversarial examples."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import imagenet
import inception_resnet_v2
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim
tf.app.flags.DEFINE_integer('batch_size', 50, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('max_num_batches', None, 'Max number of batches to evaluate by default use all.')
tf.app.flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/tfmodel/', 'The directory where the model was written to or an absolute path to a checkpoint file.')
tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4, 'The number of threads used to create the batches.')
tf.app.flags.DEFINE_string('split_name', 'validation', 'The name of the train/test split.')
tf.app.flags.DEFINE_string('dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string('model_name', 'inception_v3', 'Name of the model to use, either "inception_v3" or "inception_resnet_v2"')
tf.app.flags.DEFINE_float('moving_average_decay', None, 'The decay to use for the moving average.If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_string('adversarial_method', 'none', 'What kind of adversarial examples to use for evaluation. Could be one of: "none", "stepll", "stepllnoise".')
tf.app.flags.DEFINE_float('adversarial_eps', 0.0, 'Size of adversarial perturbation in range [0, 255].')
FLAGS = tf.app.flags.FLAGS
IMAGE_SIZE = 299
NUM_CLASSES = 1001

def preprocess_for_eval(image, height, width, central_fraction=0.875, scope=None):
    if False:
        i = 10
        return i + 15
    'Prepare one image for evaluation.\n\n  If height and width are specified it would output an image with that size by\n  applying resize_bilinear.\n  If central_fraction is specified it would crop the central fraction of the\n  input image.\n\n  Args:\n    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be\n      [0, 1], otherwise it would converted to tf.float32 assuming that the range\n      is [0, MAX], where MAX is largest positive representable number for\n      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details)\n    height: integer\n    width: integer\n    central_fraction: Optional Float, fraction of the image to crop.\n    scope: Optional scope for name_scope.\n  Returns:\n    3-D float Tensor of prepared image.\n  '
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)
        if height and width:
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image

def create_model(x, reuse=None):
    if False:
        print('Hello World!')
    'Create model graph.\n\n  Args:\n    x: input images\n    reuse: reuse parameter which will be passed to underlying variable scopes.\n      Should be None first call and True every subsequent call.\n\n  Returns:\n    (logits, end_points) - tuple of model logits and enpoints\n\n  Raises:\n    ValueError: if model type specified by --model_name flag is invalid.\n  '
    if FLAGS.model_name == 'inception_v3':
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            return inception.inception_v3(x, num_classes=NUM_CLASSES, is_training=False, reuse=reuse)
    elif FLAGS.model_name == 'inception_resnet_v2':
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            return inception_resnet_v2.inception_resnet_v2(x, num_classes=NUM_CLASSES, is_training=False, reuse=reuse)
    else:
        raise ValueError('Invalid model name: %s' % FLAGS.model_name)

def step_target_class_adversarial_images(x, eps, one_hot_target_class):
    if False:
        print('Hello World!')
    'Base code for one step towards target class methods.\n\n  Args:\n    x: source images\n    eps: size of adversarial perturbation\n    one_hot_target_class: one hot encoded target classes for all images\n\n  Returns:\n    tensor with adversarial images\n  '
    (logits, end_points) = create_model(x, reuse=True)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class, logits, label_smoothing=0.1, weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot_target_class, end_points['AuxLogits'], label_smoothing=0.1, weights=0.4)
    x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0])
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
    return tf.stop_gradient(x_adv)

def stepll_adversarial_images(x, eps):
    if False:
        print('Hello World!')
    'One step towards least likely class (Step L.L.) adversarial examples.\n\n  This method is an alternative to FGSM which does not use true classes.\n  Method is described in the "Adversarial Machine Learning at Scale" paper,\n  https://arxiv.org/abs/1611.01236\n\n  Args:\n    x: source images\n    eps: size of adversarial perturbation\n\n  Returns:\n    adversarial images\n  '
    (logits, _) = create_model(x, reuse=True)
    least_likely_class = tf.argmin(logits, 1)
    one_hot_ll_class = tf.one_hot(least_likely_class, NUM_CLASSES)
    return step_target_class_adversarial_images(x, eps, one_hot_ll_class)

def stepllnoise_adversarial_images(x, eps):
    if False:
        while True:
            i = 10
    'Step L.L. with noise method.\n\n  This is an imporvement of Step L.L. method. This method is better against\n  adversarially trained models which learn to mask gradient.\n  Method is described in the section "New randomized one shot attack" of\n  "Ensemble Adversarial Training: Attacks and Defenses" paper,\n  https://arxiv.org/abs/1705.07204\n\n  Args:\n    x: source images\n    eps: size of adversarial perturbation\n\n  Returns:\n    adversarial images\n  '
    (logits, _) = create_model(x, reuse=True)
    least_likely_class = tf.argmin(logits, 1)
    one_hot_ll_class = tf.one_hot(least_likely_class, NUM_CLASSES)
    x_noise = x + eps / 2 * tf.sign(tf.random_normal(x.shape))
    return step_target_class_adversarial_images(x_noise, eps / 2, one_hot_ll_class)

def get_input_images(dataset_images):
    if False:
        while True:
            i = 10
    'Gets input images for the evaluation.\n\n  Args:\n    dataset_images: tensor with dataset images\n\n  Returns:\n    tensor with input images, which is either dataset images or adversarial\n    images.\n\n  Raises:\n    ValueError: if adversarial method specified by --adversarial_method flag\n      is invalid.\n  '
    eps = FLAGS.adversarial_eps / 255 * 2.0
    if FLAGS.adversarial_method == 'stepll':
        return stepll_adversarial_images(dataset_images, eps)
    elif FLAGS.adversarial_method == 'stepllnoise':
        return stepllnoise_adversarial_images(dataset_images, eps)
    elif FLAGS.adversarial_method == 'none':
        return dataset_images
    else:
        raise ValueError('Invalid adversarial method: %s' % FLAGS.adversarial_method)

def main(_):
    if False:
        while True:
            i = 10
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = tf.train.get_or_create_global_step()
        dataset = imagenet.get_split(FLAGS.split_name, FLAGS.dataset_dir)
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset, shuffle=False, common_queue_capacity=2 * FLAGS.batch_size, common_queue_min=FLAGS.batch_size)
        [dataset_image, label] = provider.get(['image', 'label'])
        dataset_image = preprocess_for_eval(dataset_image, IMAGE_SIZE, IMAGE_SIZE)
        (dataset_images, labels) = tf.train.batch([dataset_image, label], batch_size=FLAGS.batch_size, num_threads=FLAGS.num_preprocessing_threads, capacity=5 * FLAGS.batch_size)
        create_model(tf.placeholder(tf.float32, shape=dataset_images.shape))
        input_images = get_input_images(dataset_images)
        (logits, _) = create_model(input_images, reuse=True)
        if FLAGS.moving_average_decay > 0:
            variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()
        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)
        (names_to_values, names_to_updates) = slim.metrics.aggregate_metric_map({'Accuracy': slim.metrics.streaming_accuracy(predictions, labels), 'Recall_5': slim.metrics.streaming_sparse_recall_at_k(logits, tf.reshape(labels, [-1, 1]), 5)})
        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
        tf.logging.info('Evaluating %s' % checkpoint_path)
        (top1_accuracy, top5_accuracy) = slim.evaluation.evaluate_once(master=FLAGS.master, checkpoint_path=checkpoint_path, logdir=None, summary_op=None, num_evals=num_batches, eval_op=list(names_to_updates.values()), final_op=[names_to_values['Accuracy'], names_to_values['Recall_5']], variables_to_restore=variables_to_restore)
        print('Top1 Accuracy: ', top1_accuracy)
        print('Top5 Accuracy: ', top5_accuracy)
if __name__ == '__main__':
    tf.app.run()