"""Helper utilities for training and testing optimizers."""
from collections import defaultdict
import random
import sys
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
from learned_optimizer.optimizer import trainable_optimizer
from learned_optimizer.optimizer import utils
from learned_optimizer.problems import datasets
from learned_optimizer.problems import problem_generator
tf.app.flags.DEFINE_integer('ps_tasks', 0, 'Number of tasks in the ps job.\n                            If 0 no ps job is used.')
tf.app.flags.DEFINE_float('nan_l2_reg', 0.01, 'Strength of l2-reg when NaNs are encountered.')
tf.app.flags.DEFINE_float('l2_reg', 0.0, 'Lambda value for parameter regularization.')
tf.app.flags.DEFINE_float('rms_decay', 0.9, 'Decay value for the RMSProp metaoptimizer.')
tf.app.flags.DEFINE_float('rms_epsilon', 1e-20, 'Epsilon value for the RMSProp metaoptimizer.')
tf.app.flags.DEFINE_boolean('set_profiling', False, 'Enable memory usage and computation time tracing for tensorflow nodes (available in TensorBoard).')
tf.app.flags.DEFINE_boolean('reset_rnn_params', True, 'Reset the parameters of the optimizer\n                               from one meta-iteration to the next.')
FLAGS = tf.app.flags.FLAGS
OPTIMIZER_SCOPE = 'LOL'
OPT_SUM_COLLECTION = 'LOL_summaries'

def sigmoid_weights(n, slope=0.1, offset=5):
    if False:
        i = 10
        return i + 15
    'Generates a sigmoid, scaled to sum to 1.\n\n  This function is used to generate weights that serve to mask out\n  the early objective values of an optimization problem such that\n  initial variation in the objective is phased out (hence the sigmoid\n  starts at zero and ramps up to the maximum value, and the total\n  weight is normalized to sum to one)\n\n  Args:\n    n: the number of samples\n    slope: slope of the sigmoid (Default: 0.1)\n    offset: threshold of the sigmoid (Default: 5)\n\n  Returns:\n    No\n  '
    x = np.arange(n)
    y = 1.0 / (1.0 + np.exp(-slope * (x - offset)))
    y_normalized = y / np.sum(y)
    return y_normalized

def sample_numiter(scale, min_steps=50):
    if False:
        i = 10
        return i + 15
    'Samples a number of iterations from an exponential distribution.\n\n  Args:\n    scale: parameter for the exponential distribution\n    min_steps: minimum number of steps to run (additive)\n\n  Returns:\n    num_steps: An integer equal to a rounded sample from the exponential\n               distribution + the value of min_steps.\n  '
    return int(np.round(np.random.exponential(scale=scale)) + min_steps)

def train_optimizer(logdir, optimizer_spec, problems_and_data, num_problems, num_meta_iterations, num_unroll_func, num_partial_unroll_itrs_func, learning_rate=0.0001, gradient_clip=5.0, is_chief=False, select_random_problems=True, callbacks=None, obj_train_max_multiplier=-1, out=sys.stdout):
    if False:
        for i in range(10):
            print('nop')
    "Trains the meta-parameters of this optimizer.\n\n  Args:\n    logdir: a directory filepath for storing model checkpoints (must exist)\n    optimizer_spec: specification for an Optimizer (see utils.Spec)\n    problems_and_data: a list of tuples containing three elements: a problem\n      specification (see utils.Spec), a dataset (see datasets.Dataset), and\n      a batch_size (int) for generating a problem and corresponding dataset. If\n      the problem doesn't have data, set dataset to None.\n    num_problems: the number of problems to sample during meta-training\n    num_meta_iterations: the number of iterations (steps) to run the\n      meta-optimizer for on each subproblem.\n    num_unroll_func: called once per meta iteration and returns the number of\n      unrolls to do for that meta iteration.\n    num_partial_unroll_itrs_func: called once per unroll and returns the number\n      of iterations to do for that unroll.\n    learning_rate: learning rate of the RMSProp meta-optimizer (Default: 1e-4)\n    gradient_clip: value to clip gradients at (Default: 5.0)\n    is_chief: whether this is the chief task (Default: False)\n    select_random_problems: whether to select training problems randomly\n        (Default: True)\n    callbacks: a list of callback functions that is run after every random\n        problem draw\n    obj_train_max_multiplier: the maximum increase in the objective value over\n        a single training run. Ignored if < 0.\n    out: where to write output to, e.g. a file handle (Default: sys.stdout)\n\n  Raises:\n    ValueError: If one of the subproblems has a negative objective value.\n  "
    if select_random_problems:
        sampler = (random.choice(problems_and_data) for _ in range(num_problems))
    else:
        num_repeats = num_problems / len(problems_and_data) + 1
        random.shuffle(problems_and_data)
        sampler = (problems_and_data * num_repeats)[:num_problems]
    for (problem_itr, (problem_spec, dataset, batch_size)) in enumerate(sampler):
        problem_start_time = time.time()
        if dataset is None:
            dataset = datasets.EMPTY_DATASET
            batch_size = dataset.size
        graph = tf.Graph()
        real_device_setter = tf.train.replica_device_setter(FLAGS.ps_tasks)

        def custom_device_setter(op):
            if False:
                while True:
                    i = 10
            if trainable_optimizer.is_local_state_variable(op):
                return '/job:worker'
            else:
                return real_device_setter(op)
        if real_device_setter:
            device_setter = custom_device_setter
        else:
            device_setter = None
        with graph.as_default(), graph.device(device_setter):
            problem = problem_spec.build()
            opt = optimizer_spec.build()
            train_output = opt.train(problem, dataset)
            state_keys = opt.state_keys
            for (key, val) in zip(state_keys, train_output.output_state[0]):
                finite_val = utils.make_finite(val, replacement=tf.zeros_like(val))
                tf.summary.histogram('State/{}'.format(key), finite_val, collections=[OPT_SUM_COLLECTION])
            tf.summary.scalar('MetaObjective', train_output.metaobj, collections=[OPT_SUM_COLLECTION])
            tf.summary.scalar(problem_spec.callable.__name__ + '_MetaObjective', train_output.metaobj, collections=[OPT_SUM_COLLECTION])
            global_step = tf.Variable(0, name='global_step', trainable=False)
            meta_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=OPTIMIZER_SCOPE)
            reg_l2 = FLAGS.l2_reg * sum([tf.reduce_sum(param ** 2) for param in meta_parameters])
            meta_opt = tf.train.RMSPropOptimizer(learning_rate, decay=FLAGS.rms_decay, use_locking=True, epsilon=FLAGS.rms_epsilon)
            grads_and_vars = meta_opt.compute_gradients(train_output.metaobj + reg_l2, meta_parameters)
            clipped_grads_and_vars = []
            for (grad, var) in grads_and_vars:
                clipped_grad = tf.clip_by_value(utils.make_finite(grad, replacement=tf.zeros_like(var)), -gradient_clip, gradient_clip)
                clipped_grads_and_vars.append((clipped_grad, var))
            for (grad, var) in grads_and_vars:
                tf.summary.histogram(var.name + '_rawgrad', utils.make_finite(grad, replacement=tf.zeros_like(grad)), collections=[OPT_SUM_COLLECTION])
            for (grad, var) in clipped_grads_and_vars:
                tf.summary.histogram(var.name + '_var', var, collections=[OPT_SUM_COLLECTION])
                tf.summary.histogram(var.name + '_grad', grad, collections=[OPT_SUM_COLLECTION])
            train_op = meta_opt.apply_gradients(clipped_grads_and_vars, global_step=global_step)
            summary_op = tf.summary.merge_all(key=OPT_SUM_COLLECTION)
            with tf.control_dependencies([train_op, summary_op]):
                propagate_loop_state_ops = []
                for (dest, src) in zip(train_output.init_loop_vars, train_output.output_loop_vars):
                    propagate_loop_state_ops.append(dest.assign(src))
                propagate_loop_state_op = tf.group(*propagate_loop_state_ops)
            sv = tf.train.Supervisor(graph=graph, is_chief=is_chief, logdir=logdir, summary_op=None, save_model_secs=0, global_step=global_step)
            with sv.managed_session() as sess:
                init_time = time.time() - problem_start_time
                out.write('--------- Problem #{} ---------\n'.format(problem_itr))
                out.write('{callable.__name__}{args}{kwargs}\n'.format(**problem_spec.__dict__))
                out.write('Took {} seconds to initialize.\n'.format(init_time))
                out.flush()
                if FLAGS.set_profiling:
                    summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)
                metadata = defaultdict(list)
                for k in range(num_meta_iterations):
                    if sv.should_stop():
                        break
                    problem.init_fn(sess)
                    full_trace_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_options = full_trace_opt if FLAGS.set_profiling else None
                    run_metadata = tf.RunMetadata() if FLAGS.set_profiling else None
                    num_unrolls = num_unroll_func()
                    partial_unroll_iters = [num_partial_unroll_itrs_func() for _ in xrange(num_unrolls)]
                    total_num_iter = sum(partial_unroll_iters)
                    objective_weights = [np.ones(num) / float(num) for num in partial_unroll_iters]
                    db = dataset.batch_indices(total_num_iter, batch_size)
                    dataset_batches = []
                    last_index = 0
                    for num in partial_unroll_iters:
                        dataset_batches.append(db[last_index:last_index + num])
                        last_index += num
                    train_start_time = time.time()
                    unroll_itr = 0
                    additional_log_info = ''
                    for unroll_itr in range(num_unrolls):
                        first_unroll = unroll_itr == 0
                        if FLAGS.reset_rnn_params:
                            reset_state = first_unroll and k == 0
                        else:
                            reset_state = first_unroll
                        feed = {train_output.obj_weights: objective_weights[unroll_itr], train_output.batches: dataset_batches[unroll_itr], train_output.first_unroll: first_unroll, train_output.reset_state: reset_state}
                        fetches_list = [train_output.metaobj, train_output.problem_objectives, train_output.initial_obj, summary_op, clipped_grads_and_vars, train_op]
                        if unroll_itr + 1 < num_unrolls:
                            fetches_list += [propagate_loop_state_op]
                        fetched = sess.run(fetches_list, feed_dict=feed, options=run_options, run_metadata=run_metadata)
                        meta_obj = fetched[0]
                        sub_obj = fetched[1]
                        init_obj = fetched[2]
                        summ = fetched[3]
                        meta_grads_and_params = fetched[4]
                        if np.any(sub_obj < 0):
                            raise ValueError('Training problem objectives must be nonnegative.')
                        if obj_train_max_multiplier > 0 and sub_obj[-1] > init_obj + abs(init_obj) * (obj_train_max_multiplier - 1):
                            msg = ' Broke early at {} out of {} unrolls. '.format(unroll_itr + 1, num_unrolls)
                            additional_log_info += msg
                            break
                        if is_chief:
                            sv.summary_computed(sess, summ)
                        metadata['subproblem_objs'].append(sub_obj)
                        metadata['meta_objs'].append(meta_obj)
                        metadata['meta_grads_and_params'].append(meta_grads_and_params)
                    optimization_time = time.time() - train_start_time
                    if FLAGS.set_profiling:
                        summary_name = '%02d_iter%04d_%02d' % (FLAGS.task, problem_itr, k)
                        summary_writer.add_run_metadata(run_metadata, summary_name)
                    metadata['global_step'].append(sess.run(global_step))
                    metadata['runtimes'].append(optimization_time)
                    args = (k, meta_obj, optimization_time, sum(partial_unroll_iters[:unroll_itr + 1]))
                    out.write('  [{:02}] {}, {} seconds, {} iters '.format(*args))
                    out.write('(unrolled {} steps)'.format(', '.join([str(s) for s in partial_unroll_iters[:unroll_itr + 1]])))
                    out.write('{}\n'.format(additional_log_info))
                    out.flush()
                if FLAGS.set_profiling:
                    summary_writer.close()
                if is_chief:
                    sv.saver.save(sess, sv.save_path, global_step=global_step)
        if is_chief and callbacks is not None:
            for callback in callbacks:
                if hasattr(callback, '__call__'):
                    problem_name = problem_spec.callable.__name__
                    callback(problem_name, problem_itr, logdir, metadata)

def test_optimizer(optimizer, problem, num_iter, dataset=datasets.EMPTY_DATASET, batch_size=None, seed=None, graph=None, logdir=None, record_every=None):
    if False:
        print('Hello World!')
    'Tests an optimization algorithm on a given problem.\n\n  Args:\n    optimizer: Either a tf.train.Optimizer instance, or an Optimizer instance\n               inheriting from trainable_optimizer.py\n    problem: A Problem instance that defines an optimization problem to solve\n    num_iter: The number of iterations of the optimizer to run\n    dataset: The dataset to train the problem against\n    batch_size: The number of samples per batch. If None (default), the\n      batch size is set to the full batch (dataset.size)\n    seed: A random seed used for drawing the initial parameters, or a list of\n      numpy arrays used to explicitly initialize the parameters.\n    graph: The tensorflow graph to execute (if None, uses the default graph)\n    logdir: A directory containing model checkpoints. If given, then the\n            parameters of the optimizer are loaded from the latest checkpoint\n            in this folder.\n    record_every: if an integer, stores the parameters, objective, and gradient\n                  every recored_every iterations. If None, nothing is stored\n\n  Returns:\n    objective_values: A list of the objective values during optimization\n    parameters: The parameters obtained after training\n    records: A dictionary containing lists of the parameters and gradients\n             during optimization saved every record_every iterations (empty if\n             record_every is set to None)\n  '
    if dataset is None:
        dataset = datasets.EMPTY_DATASET
        batch_size = dataset.size
    else:
        batch_size = dataset.size if batch_size is None else batch_size
    graph = tf.get_default_graph() if graph is None else graph
    with graph.as_default():
        if isinstance(seed, (list, tuple)):
            params = problem_generator.init_fixed_variables(seed)
        else:
            params = problem.init_variables(seed)
        data_placeholder = tf.placeholder(tf.float32)
        labels_placeholder = tf.placeholder(tf.int32)
        obj = problem.objective(params, data_placeholder, labels_placeholder)
        gradients = problem.gradients(obj, params)
        vars_to_preinitialize = params
    with tf.Session(graph=graph) as sess:
        sess.run(tf.variables_initializer(vars_to_preinitialize))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            (train_op, real_params) = optimizer.apply_gradients(zip(gradients, params))
            obj = problem.objective(real_params, data_placeholder, labels_placeholder)
        except TypeError:
            train_op = optimizer.apply_gradients(zip(gradients, params))
        vars_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=OPTIMIZER_SCOPE)
        vars_to_initialize = list(set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) - set(vars_to_restore) - set(vars_to_preinitialize))
        if logdir is not None:
            restorer = tf.Saver(var_list=vars_to_restore)
            ckpt = tf.train.latest_checkpoint(logdir)
            restorer.restore(sess, ckpt)
        else:
            sess.run(tf.variables_initializer(vars_to_restore))
        sess.run(tf.variables_initializer(vars_to_initialize))
        problem.init_fn(sess)
        batch_inds = dataset.batch_indices(num_iter, batch_size)
        records = defaultdict(list)
        objective_values = []
        for (itr, batch) in enumerate(batch_inds):
            feed = {data_placeholder: dataset.data[batch], labels_placeholder: dataset.labels[batch]}
            full_feed = {data_placeholder: dataset.data, labels_placeholder: dataset.labels}
            if record_every is not None and itr % record_every == 0:

                def grad_value(g):
                    if False:
                        for i in range(10):
                            print('nop')
                    if isinstance(g, tf.IndexedSlices):
                        return g.values
                    else:
                        return g
                records_fetch = {}
                for p in params:
                    for key in optimizer.get_slot_names():
                        v = optimizer.get_slot(p, key)
                        records_fetch[p.name + '_' + key] = v
                gav_fetch = [(grad_value(g), v) for (g, v) in zip(gradients, params)]
                (_, gav_eval, records_eval) = sess.run((obj, gav_fetch, records_fetch), feed_dict=feed)
                full_obj_eval = sess.run([obj], feed_dict=full_feed)
                records['objective'].append(full_obj_eval)
                records['grad_norm'].append([np.linalg.norm(g.ravel()) for (g, _) in gav_eval])
                records['param_norm'].append([np.linalg.norm(v.ravel()) for (_, v) in gav_eval])
                records['grad'].append([g for (g, _) in gav_eval])
                records['param'].append([v for (_, v) in gav_eval])
                records['iter'].append(itr)
                for (k, v) in records_eval.iteritems():
                    records[k].append(v)
            objective_values.append(sess.run([train_op, obj], feed_dict=feed)[1])
        parameters = [sess.run(p) for p in params]
        coord.request_stop()
        coord.join(threads)
    return (objective_values, parameters, records)

def run_wall_clock_test(optimizer, problem, num_steps, dataset=datasets.EMPTY_DATASET, seed=None, logdir=None, batch_size=None):
    if False:
        return 10
    'Runs optimization with the given parameters and return average iter time.\n\n  Args:\n    optimizer: The tf.train.Optimizer instance\n    problem: The problem to optimize (a problem_generator.Problem)\n    num_steps: The number of steps to run optimization for\n    dataset: The dataset to train the problem against\n    seed: The seed used for drawing the initial parameters, or a list of\n      numpy arrays used to explicitly initialize the parameters\n    logdir: A directory containing model checkpoints. If given, then the\n            parameters of the optimizer are loaded from the latest checkpoint\n            in this folder.\n    batch_size: The number of samples per batch.\n\n  Returns:\n    The average time in seconds for a single optimization iteration.\n  '
    if dataset is None:
        dataset = datasets.EMPTY_DATASET
        batch_size = dataset.size
    else:
        batch_size = dataset.size if batch_size is None else batch_size
    if isinstance(seed, (list, tuple)):
        params = problem_generator.init_fixed_variables(seed)
    else:
        params = problem.init_variables(seed)
    data_placeholder = tf.placeholder(tf.float32)
    labels_placeholder = tf.placeholder(tf.int32)
    obj = problem.objective(params, data_placeholder, labels_placeholder)
    gradients = problem.gradients(obj, params)
    vars_to_preinitialize = params
    with tf.Session(graph=tf.get_default_graph()) as sess:
        sess.run(tf.variables_initializer(vars_to_preinitialize))
        train_op = optimizer.apply_gradients(zip(gradients, params))
        if isinstance(train_op, tuple) or isinstance(train_op, list):
            train_op = train_op[0]
        vars_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=OPTIMIZER_SCOPE)
        vars_to_initialize = list(set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) - set(vars_to_restore) - set(vars_to_preinitialize))
        if logdir is not None:
            restorer = tf.Saver(var_list=vars_to_restore)
            ckpt = tf.train.latest_checkpoint(logdir)
            restorer.restore(sess, ckpt)
        else:
            sess.run(tf.variables_initializer(vars_to_restore))
        sess.run(tf.variables_initializer(vars_to_initialize))
        problem.init_fn(sess)
        batch_inds = dataset.batch_indices(num_steps, batch_size)
        avg_iter_time = []
        for batch in batch_inds:
            feed = {data_placeholder: dataset.data[batch], labels_placeholder: dataset.labels[batch]}
            start = time.time()
            sess.run([train_op], feed_dict=feed)
            avg_iter_time.append(time.time() - start)
    return np.median(np.array(avg_iter_time))