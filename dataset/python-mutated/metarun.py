"""Scripts for meta-optimization."""
from __future__ import print_function
import os
import tensorflow as tf
import metaopt
from learned_optimizer.optimizer import coordinatewise_rnn
from learned_optimizer.optimizer import global_learning_rate
from learned_optimizer.optimizer import hierarchical_rnn
from learned_optimizer.optimizer import learning_rate_schedule
from learned_optimizer.optimizer import trainable_adam
from learned_optimizer.problems import problem_sets as ps
from learned_optimizer.problems import problem_spec
tf.app.flags.DEFINE_string('train_dir', '/tmp/lol/', 'Directory to store parameters and results.')
tf.app.flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')
tf.app.flags.DEFINE_integer('worker_tasks', 1, 'Number of tasks in the worker job.')
tf.app.flags.DEFINE_integer('num_problems', 1000, 'Number of sub-problems to run.')
tf.app.flags.DEFINE_integer('num_meta_iterations', 5, 'Number of meta-iterations to optimize.')
tf.app.flags.DEFINE_integer('num_unroll_scale', 40, 'The scale parameter of the exponential\n                            distribution from which the number of partial\n                            unrolls is drawn')
tf.app.flags.DEFINE_integer('min_num_unrolls', 1, 'The minimum number of unrolls per problem.')
tf.app.flags.DEFINE_integer('num_partial_unroll_itr_scale', 200, 'The scale parameter of the exponential\n                               distribution from which the number of iterations\n                               per unroll is drawn.')
tf.app.flags.DEFINE_integer('min_num_itr_partial_unroll', 50, 'The minimum number of iterations for one\n                               unroll.')
tf.app.flags.DEFINE_string('optimizer', 'HierarchicalRNN', 'Which meta-optimizer to train.')
tf.app.flags.DEFINE_integer('cell_size', 20, 'Size of the RNN hidden state in each layer.')
tf.app.flags.DEFINE_integer('num_cells', 2, 'Number of RNN layers.')
tf.app.flags.DEFINE_string('cell_cls', 'GRUCell', 'Type of RNN cell to use.')
tf.app.flags.DEFINE_float('meta_learning_rate', 1e-06, 'The learning rate for the meta-optimizer.')
tf.app.flags.DEFINE_float('gradient_clip_level', 10000.0, 'The level to clip gradients to.')
tf.app.flags.DEFINE_boolean('include_quadratic_problems', False, 'Include non-noisy quadratic problems.')
tf.app.flags.DEFINE_boolean('include_noisy_quadratic_problems', True, 'Include noisy quadratic problems.')
tf.app.flags.DEFINE_boolean('include_large_quadratic_problems', True, 'Include very large quadratic problems.')
tf.app.flags.DEFINE_boolean('include_bowl_problems', True, 'Include 2D bowl problems.')
tf.app.flags.DEFINE_boolean('include_softmax_2_class_problems', True, 'Include 2-class logistic regression problems.')
tf.app.flags.DEFINE_boolean('include_noisy_softmax_2_class_problems', True, 'Include noisy 2-class logistic regression\n                               problems.')
tf.app.flags.DEFINE_boolean('include_optimization_test_problems', True, 'Include non-noisy versions of classic\n                               optimization test problems, e.g. Rosenbrock.')
tf.app.flags.DEFINE_boolean('include_noisy_optimization_test_problems', True, 'Include gradient-noise versions of classic\n                               optimization test problems, e.g. Rosenbrock')
tf.app.flags.DEFINE_boolean('include_fully_connected_random_2_class_problems', True, 'Include MLP problems for 2 classes.')
tf.app.flags.DEFINE_boolean('include_matmul_problems', True, 'Include matrix multiplication problems.')
tf.app.flags.DEFINE_boolean('include_log_objective_problems', True, 'Include problems where the objective is the log\n                               objective of another problem, e.g. Bowl.')
tf.app.flags.DEFINE_boolean('include_rescale_problems', True, 'Include problems where the parameters are scaled\n                               version of the original parameters.')
tf.app.flags.DEFINE_boolean('include_norm_problems', True, 'Include problems where the objective is the\n                               N-norm of another problem, e.g. Quadratic.')
tf.app.flags.DEFINE_boolean('include_sum_problems', True, 'Include problems where the objective is the sum\n                               of the objectives of the subproblems that make\n                               up the problem parameters. Per-problem tensors\n                               are still independent of each other.')
tf.app.flags.DEFINE_boolean('include_sparse_gradient_problems', True, 'Include problems where the gradient is set to 0\n                               with some high probability.')
tf.app.flags.DEFINE_boolean('include_sparse_softmax_problems', False, 'Include sparse softmax problems.')
tf.app.flags.DEFINE_boolean('include_one_hot_sparse_softmax_problems', False, 'Include one-hot sparse softmax problems.')
tf.app.flags.DEFINE_boolean('include_noisy_bowl_problems', True, 'Include noisy bowl problems.')
tf.app.flags.DEFINE_boolean('include_noisy_norm_problems', True, 'Include noisy norm problems.')
tf.app.flags.DEFINE_boolean('include_noisy_sum_problems', True, 'Include noisy sum problems.')
tf.app.flags.DEFINE_boolean('include_sum_of_quadratics_problems', False, 'Include sum of quadratics problems.')
tf.app.flags.DEFINE_boolean('include_projection_quadratic_problems', False, 'Include projection quadratic problems.')
tf.app.flags.DEFINE_boolean('include_outward_snake_problems', False, 'Include outward snake problems.')
tf.app.flags.DEFINE_boolean('include_dependency_chain_problems', False, 'Include dependency chain problems.')
tf.app.flags.DEFINE_boolean('include_min_max_well_problems', False, 'Include min-max well problems.')
tf.app.flags.DEFINE_float('min_lr', 1e-06, 'The minimum initial learning rate.')
tf.app.flags.DEFINE_float('max_lr', 0.01, 'The maximum initial learning rate.')
tf.app.flags.DEFINE_boolean('zero_init_lr_weights', True, 'Whether to initialize the learning rate weights\n                               to 0 rather than the scaled random initialization\n                               used for other RNN variables.')
tf.app.flags.DEFINE_boolean('use_relative_lr', True, 'Whether to use the relative learning rate as an\n                               input during training. Can only be used if\n                               learnable_decay is also True.')
tf.app.flags.DEFINE_boolean('use_extreme_indicator', False, 'Whether to use the extreme indicator for learning\n                               rates as an input during training. Can only be\n                               used if learnable_decay is also True.')
tf.app.flags.DEFINE_boolean('use_log_means_squared', True, 'Whether to track the log of the mean squared\n                               grads instead of the means squared grads.')
tf.app.flags.DEFINE_boolean('use_problem_lr_mean', True, 'Whether to use the mean over all learning rates\n                               in the problem when calculating the relative\n                               learning rate.')
tf.app.flags.DEFINE_boolean('learnable_decay', True, 'Whether to learn weights that dynamically\n                              modulate the input scale via RMS decay.')
tf.app.flags.DEFINE_boolean('dynamic_output_scale', True, 'Whether to learn weights that dynamically\n                               modulate the output scale.')
tf.app.flags.DEFINE_boolean('use_log_objective', True, 'Whether to use the log of the scaled objective\n                               rather than just the scaled obj for training.')
tf.app.flags.DEFINE_boolean('use_attention', False, 'Whether to learn where to attend.')
tf.app.flags.DEFINE_boolean('use_second_derivatives', True, 'Whether to use second derivatives.')
tf.app.flags.DEFINE_integer('num_gradient_scales', 4, 'How many different timescales to keep for\n                               gradient history. If > 1, also learns a scale\n                               factor for gradient history.')
tf.app.flags.DEFINE_float('max_log_lr', 33, 'The maximum log learning rate allowed.')
tf.app.flags.DEFINE_float('objective_training_max_multiplier', -1, 'How much the objective can grow before training on\n                             this problem / param pair is terminated. Sets a max\n                             on the objective value when multiplied by the\n                             initial objective. If <= 0, not used.')
tf.app.flags.DEFINE_boolean('use_gradient_shortcut', True, 'Whether to add a learned affine projection of the\n                               gradient to the update delta in addition to the\n                               gradient function computed by the RNN.')
tf.app.flags.DEFINE_boolean('use_lr_shortcut', False, 'Whether to add the difference between the current\n                               learning rate and the desired learning rate to\n                               the RNN input.')
tf.app.flags.DEFINE_boolean('use_grad_products', True, 'Whether to use gradient products in the input to\n                               the RNN. Only applicable when num_gradient_scales\n                               > 1.')
tf.app.flags.DEFINE_boolean('use_multiple_scale_decays', False, 'Whether to use many-timescale scale decays.')
tf.app.flags.DEFINE_boolean('use_numerator_epsilon', False, 'Whether to use epsilon in the numerator of the\n                               log objective.')
tf.app.flags.DEFINE_boolean('learnable_inp_decay', True, 'Whether to learn input decay weight and bias.')
tf.app.flags.DEFINE_boolean('learnable_rnn_init', True, 'Whether to learn RNN state initialization.')
FLAGS = tf.app.flags.FLAGS
HRNN_CELL_SIZES = [10, 20, 20]

def register_optimizers():
    if False:
        i = 10
        return i + 15
    opts = {}
    opts['CoordinatewiseRNN'] = coordinatewise_rnn.CoordinatewiseRNN
    opts['GlobalLearningRate'] = global_learning_rate.GlobalLearningRate
    opts['HierarchicalRNN'] = hierarchical_rnn.HierarchicalRNN
    opts['LearningRateSchedule'] = learning_rate_schedule.LearningRateSchedule
    opts['TrainableAdam'] = trainable_adam.TrainableAdam
    return opts

def main(unused_argv):
    if False:
        i = 10
        return i + 15
    'Runs the main script.'
    opts = register_optimizers()
    problems_and_data = []
    if FLAGS.include_sparse_softmax_problems:
        problems_and_data.extend(ps.sparse_softmax_2_class_sparse_problems())
    if FLAGS.include_one_hot_sparse_softmax_problems:
        problems_and_data.extend(ps.one_hot_sparse_softmax_2_class_sparse_problems())
    if FLAGS.include_quadratic_problems:
        problems_and_data.extend(ps.quadratic_problems())
    if FLAGS.include_noisy_quadratic_problems:
        problems_and_data.extend(ps.quadratic_problems_noisy())
    if FLAGS.include_large_quadratic_problems:
        problems_and_data.extend(ps.quadratic_problems_large())
    if FLAGS.include_bowl_problems:
        problems_and_data.extend(ps.bowl_problems())
    if FLAGS.include_noisy_bowl_problems:
        problems_and_data.extend(ps.bowl_problems_noisy())
    if FLAGS.include_softmax_2_class_problems:
        problems_and_data.extend(ps.softmax_2_class_problems())
    if FLAGS.include_noisy_softmax_2_class_problems:
        problems_and_data.extend(ps.softmax_2_class_problems_noisy())
    if FLAGS.include_optimization_test_problems:
        problems_and_data.extend(ps.optimization_test_problems())
    if FLAGS.include_noisy_optimization_test_problems:
        problems_and_data.extend(ps.optimization_test_problems_noisy())
    if FLAGS.include_fully_connected_random_2_class_problems:
        problems_and_data.extend(ps.fully_connected_random_2_class_problems())
    if FLAGS.include_matmul_problems:
        problems_and_data.extend(ps.matmul_problems())
    if FLAGS.include_log_objective_problems:
        problems_and_data.extend(ps.log_objective_problems())
    if FLAGS.include_rescale_problems:
        problems_and_data.extend(ps.rescale_problems())
    if FLAGS.include_norm_problems:
        problems_and_data.extend(ps.norm_problems())
    if FLAGS.include_noisy_norm_problems:
        problems_and_data.extend(ps.norm_problems_noisy())
    if FLAGS.include_sum_problems:
        problems_and_data.extend(ps.sum_problems())
    if FLAGS.include_noisy_sum_problems:
        problems_and_data.extend(ps.sum_problems_noisy())
    if FLAGS.include_sparse_gradient_problems:
        problems_and_data.extend(ps.sparse_gradient_problems())
        if FLAGS.include_fully_connected_random_2_class_problems:
            problems_and_data.extend(ps.sparse_gradient_problems_mlp())
    if FLAGS.include_min_max_well_problems:
        problems_and_data.extend(ps.min_max_well_problems())
    if FLAGS.include_sum_of_quadratics_problems:
        problems_and_data.extend(ps.sum_of_quadratics_problems())
    if FLAGS.include_projection_quadratic_problems:
        problems_and_data.extend(ps.projection_quadratic_problems())
    if FLAGS.include_outward_snake_problems:
        problems_and_data.extend(ps.outward_snake_problems())
    if FLAGS.include_dependency_chain_problems:
        problems_and_data.extend(ps.dependency_chain_problems())
    logdir = os.path.join(FLAGS.train_dir, '{}_{}_{}_{}'.format(FLAGS.optimizer, FLAGS.cell_cls, FLAGS.cell_size, FLAGS.num_cells))
    optimizer_cls = opts[FLAGS.optimizer]
    assert len(HRNN_CELL_SIZES) in [1, 2, 3]
    optimizer_args = (HRNN_CELL_SIZES,)
    optimizer_kwargs = {'init_lr_range': (FLAGS.min_lr, FLAGS.max_lr), 'learnable_decay': FLAGS.learnable_decay, 'dynamic_output_scale': FLAGS.dynamic_output_scale, 'cell_cls': getattr(tf.contrib.rnn, FLAGS.cell_cls), 'use_attention': FLAGS.use_attention, 'use_log_objective': FLAGS.use_log_objective, 'num_gradient_scales': FLAGS.num_gradient_scales, 'zero_init_lr_weights': FLAGS.zero_init_lr_weights, 'use_log_means_squared': FLAGS.use_log_means_squared, 'use_relative_lr': FLAGS.use_relative_lr, 'use_extreme_indicator': FLAGS.use_extreme_indicator, 'max_log_lr': FLAGS.max_log_lr, 'obj_train_max_multiplier': FLAGS.objective_training_max_multiplier, 'use_problem_lr_mean': FLAGS.use_problem_lr_mean, 'use_gradient_shortcut': FLAGS.use_gradient_shortcut, 'use_second_derivatives': FLAGS.use_second_derivatives, 'use_lr_shortcut': FLAGS.use_lr_shortcut, 'use_grad_products': FLAGS.use_grad_products, 'use_multiple_scale_decays': FLAGS.use_multiple_scale_decays, 'use_numerator_epsilon': FLAGS.use_numerator_epsilon, 'learnable_inp_decay': FLAGS.learnable_inp_decay, 'learnable_rnn_init': FLAGS.learnable_rnn_init}
    optimizer_spec = problem_spec.Spec(optimizer_cls, optimizer_args, optimizer_kwargs)
    tf.gfile.MakeDirs(logdir)
    is_chief = FLAGS.task == 0
    select_random_problems = FLAGS.worker_tasks == 1 or not is_chief

    def num_unrolls():
        if False:
            for i in range(10):
                print('nop')
        return metaopt.sample_numiter(FLAGS.num_unroll_scale, FLAGS.min_num_unrolls)

    def num_partial_unroll_itrs():
        if False:
            print('Hello World!')
        return metaopt.sample_numiter(FLAGS.num_partial_unroll_itr_scale, FLAGS.min_num_itr_partial_unroll)
    metaopt.train_optimizer(logdir, optimizer_spec, problems_and_data, FLAGS.num_problems, FLAGS.num_meta_iterations, num_unrolls, num_partial_unroll_itrs, learning_rate=FLAGS.meta_learning_rate, gradient_clip=FLAGS.gradient_clip_level, is_chief=is_chief, select_random_problems=select_random_problems, obj_train_max_multiplier=FLAGS.objective_training_max_multiplier, callbacks=[])
    return 0
if __name__ == '__main__':
    tf.app.run()