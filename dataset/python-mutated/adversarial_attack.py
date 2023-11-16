"""Library with adversarial attacks.

This library designed to be self-contained and have no dependencies other
than TensorFlow. It only contains PGD / Iterative FGSM attacks,
see https://arxiv.org/abs/1706.06083 and https://arxiv.org/abs/1607.02533
for details.

For wider set of adversarial attacks refer to Cleverhans library:
https://github.com/tensorflow/cleverhans
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def generate_pgd_common(x, bounds, model_fn, attack_params, one_hot_labels, perturbation_multiplier):
    if False:
        i = 10
        return i + 15
    'Common code for generating PGD adversarial examples.\n\n  Args:\n    x: original examples.\n    bounds: tuple with bounds of image values, bounds[0] < bounds[1].\n    model_fn: model function with signature model_fn(images).\n    attack_params: parameters of the attack.\n    one_hot_labels: one hot label vector to use in the loss.\n    perturbation_multiplier: multiplier of adversarial perturbation,\n      either +1.0 or -1.0.\n\n  Returns:\n    Tensor with adversarial examples.\n\n  Raises:\n    ValueError: if attack parameters are invalid.\n  '
    params_list = attack_params.split('_')
    if len(params_list) != 3:
        raise ValueError('Invalid parameters of PGD attack: %s' % attack_params)
    epsilon = int(params_list[0])
    step_size = int(params_list[1])
    niter = int(params_list[2])
    epsilon = float(epsilon) / 255.0 * (bounds[1] - bounds[0])
    step_size = float(step_size) / 255.0 * (bounds[1] - bounds[0])
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])
    start_x = x + tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)
    loop_vars = [0, start_x]

    def loop_cond(index, _):
        if False:
            return 10
        return index < niter

    def loop_body(index, adv_images):
        if False:
            return 10
        logits = model_fn(adv_images)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=logits))
        perturbation = step_size * tf.sign(tf.gradients(loss, adv_images)[0])
        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return (index + 1, new_adv_images)
    with tf.control_dependencies([start_x]):
        (_, result) = tf.while_loop(loop_cond, loop_body, loop_vars, back_prop=False, parallel_iterations=1)
        return result

def generate_pgd_ll(x, bounds, model_fn, attack_params):
    if False:
        while True:
            i = 10
    'Generats targeted PGD adversarial examples with least likely target class.\n\n  See generate_pgd_common for description of arguments.\n\n  Returns:\n    Tensor with adversarial examples.\n  '
    logits = model_fn(x)
    num_classes = tf.shape(logits)[1]
    one_hot_labels = tf.one_hot(tf.argmin(model_fn(x), axis=1), num_classes)
    return generate_pgd_common(x, bounds, model_fn, attack_params, one_hot_labels=one_hot_labels, perturbation_multiplier=-1.0)

def generate_pgd_rand(x, bounds, model_fn, attack_params):
    if False:
        for i in range(10):
            print('nop')
    'Generats targeted PGD adversarial examples with random target class.\n\n  See generate_pgd_common for description of arguments.\n\n  Returns:\n    Tensor with adversarial examples.\n  '
    logits = model_fn(x)
    batch_size = tf.shape(logits)[0]
    num_classes = tf.shape(logits)[1]
    random_labels = tf.random_uniform(shape=[batch_size], minval=0, maxval=num_classes, dtype=tf.int32)
    one_hot_labels = tf.one_hot(random_labels, num_classes)
    return generate_pgd_common(x, bounds, model_fn, attack_params, one_hot_labels=one_hot_labels, perturbation_multiplier=-1.0)

def generate_pgd(x, bounds, model_fn, attack_params):
    if False:
        print('Hello World!')
    'Generats non-targeted PGD adversarial examples.\n\n  See generate_pgd_common for description of arguments.\n\n  Returns:\n    tensor with adversarial examples.\n  '
    logits = model_fn(x)
    num_classes = tf.shape(logits)[1]
    one_hot_labels = tf.one_hot(tf.argmax(model_fn(x), axis=1), num_classes)
    return generate_pgd_common(x, bounds, model_fn, attack_params, one_hot_labels=one_hot_labels, perturbation_multiplier=1.0)

def generate_adversarial_examples(x, bounds, model_fn, attack_description):
    if False:
        return 10
    'Generates adversarial examples.\n\n  Args:\n    x: original examples.\n    bounds: tuple with bounds of image values, bounds[0] < bounds[1]\n    model_fn: model function with signature model_fn(images).\n    attack_description: string which describes an attack, see notes below for\n      details.\n\n  Returns:\n    Tensor with adversarial examples.\n\n  Raises:\n    ValueError: if attack description is invalid.\n\n\n  Attack description could be one of the following strings:\n  - "clean" - no attack, return original images.\n  - "pgd_EPS_STEP_NITER" - non-targeted PGD attack.\n  - "pgdll_EPS_STEP_NITER" - tageted PGD attack with least likely target class.\n  - "pgdrnd_EPS_STEP_NITER" - targetd PGD attack with random target class.\n\n  Meaning of attack parameters is following:\n  - EPS - maximum size of adversarial perturbation, between 0 and 255.\n  - STEP - step size of one iteration of PGD, between 0 and 255.\n  - NITER - number of iterations.\n  '
    if attack_description == 'clean':
        return x
    idx = attack_description.find('_')
    if idx < 0:
        raise ValueError('Invalid value of attack description %s' % attack_description)
    attack_name = attack_description[:idx]
    attack_params = attack_description[idx + 1:]
    if attack_name == 'pgdll':
        return generate_pgd_ll(x, bounds, model_fn, attack_params)
    elif attack_name == 'pgdrnd':
        return generate_pgd_rand(x, bounds, model_fn, attack_params)
    elif attack_name == 'pgd':
        return generate_pgd(x, bounds, model_fn, attack_params)
    else:
        raise ValueError('Invalid value of attack description %s' % attack_description)