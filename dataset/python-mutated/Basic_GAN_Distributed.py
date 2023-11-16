import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import cntk as C
import cntk.tests.test_utils
from timeit import default_timer as timer
cntk.tests.test_utils.set_device_from_pytest_env()
C.cntk_py.set_fixed_random_seed(1)
isFast = True
g_input_dim = 100
g_hidden_dim = 128
g_output_dim = d_input_dim = 784
d_hidden_dim = 128
d_output_dim = 1

def create_reader(path, is_training, input_dim, label_dim):
    if False:
        i = 10
        return i + 15
    deserializer = C.io.CTFDeserializer(filename=path, streams=C.io.StreamDefs(labels_unused=C.io.StreamDef(field='labels', shape=label_dim, is_sparse=False), features=C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)))
    return C.io.MinibatchSource(deserializers=deserializer, randomize=is_training, max_sweeps=C.io.INFINITELY_REPEAT if is_training else 1)
np.random.seed(123)

def noise_sample(num_samples):
    if False:
        while True:
            i = 10
    return np.random.uniform(low=-1.0, high=1.0, size=[num_samples, g_input_dim]).astype(np.float32)

def generator(z):
    if False:
        print('Hello World!')
    with C.layers.default_options(init=C.xavier()):
        h1 = C.layers.Dense(g_hidden_dim, activation=C.relu)(z)
        return C.layers.Dense(g_output_dim, activation=C.tanh)(h1)

def discriminator(x):
    if False:
        i = 10
        return i + 15
    with C.layers.default_options(init=C.xavier()):
        h1 = C.layers.Dense(d_hidden_dim, activation=C.relu)(x)
        return C.layers.Dense(d_output_dim, activation=C.sigmoid)(h1)
minibatch_size = 1024
num_minibatches = 300 if isFast else 40000
lr = 5e-05

def build_graph(noise_shape, image_shape, G_progress_printer, D_progress_printer):
    if False:
        return 10
    input_dynamic_axes = [C.Axis.default_batch_axis()]
    Z = C.input_variable(noise_shape, dynamic_axes=input_dynamic_axes)
    X_real = C.input_variable(image_shape, dynamic_axes=input_dynamic_axes)
    X_real_scaled = 2 * (X_real / 255.0) - 1.0
    X_fake = generator(Z)
    D_real = discriminator(X_real_scaled)
    D_fake = D_real.clone(method='share', substitutions={X_real_scaled.output: X_fake.output})
    G_loss = 1.0 - C.log(D_fake)
    D_loss = -(C.log(D_real) + C.log(1.0 - D_fake))
    G_learner = C.fsadagrad(parameters=X_fake.parameters, lr=C.learning_parameter_schedule_per_sample(lr), momentum=C.momentum_schedule_per_sample(0.9985724484938566))
    D_learner = C.fsadagrad(parameters=D_real.parameters, lr=C.learning_parameter_schedule_per_sample(lr), momentum=C.momentum_schedule_per_sample(0.9985724484938566))
    DistG_learner = C.train.distributed.data_parallel_distributed_learner(G_learner)
    DistD_learner = C.train.distributed.data_parallel_distributed_learner(D_learner)
    G_trainer = C.Trainer(X_fake, (G_loss, None), DistG_learner, G_progress_printer)
    D_trainer = C.Trainer(D_real, (D_loss, None), DistD_learner, D_progress_printer)
    return (X_real, X_fake, Z, G_trainer, D_trainer)

def train(reader_train):
    if False:
        for i in range(10):
            print('nop')
    k = 2
    worker_rank = C.Communicator.rank()
    print_frequency_mbsize = num_minibatches // 50
    pp_G = C.logging.ProgressPrinter(print_frequency_mbsize, rank=worker_rank)
    pp_D = C.logging.ProgressPrinter(print_frequency_mbsize * k, rank=worker_rank)
    (X_real, X_fake, Z, G_trainer, D_trainer) = build_graph(g_input_dim, d_input_dim, pp_G, pp_D)
    input_map = {X_real: reader_train.streams.features}
    num_partitions = C.Communicator.num_workers()
    worker_rank = C.Communicator.rank()
    distributed_minibatch_size = minibatch_size // num_partitions
    for train_step in range(num_minibatches):
        for gen_train_step in range(k):
            Z_data = noise_sample(distributed_minibatch_size)
            X_data = reader_train.next_minibatch(minibatch_size, input_map, num_data_partitions=num_partitions, partition_index=worker_rank)
            if X_data[X_real].num_samples == Z_data.shape[0]:
                batch_inputs = {X_real: X_data[X_real].data, Z: Z_data}
                D_trainer.train_minibatch(batch_inputs)
        Z_data = noise_sample(distributed_minibatch_size)
        batch_inputs = {Z: Z_data}
        G_trainer.train_minibatch(batch_inputs)
        G_trainer_loss = G_trainer.previous_minibatch_loss_average
    return (Z, X_fake, G_trainer_loss)

def plot_images(images, subplot_shape):
    if False:
        for i in range(10):
            print('nop')
    plt.style.use('ggplot')
    (fig, axes) = plt.subplots(*subplot_shape)
    for (image, ax) in zip(images, axes.flatten()):
        ax.imshow(image.reshape(28, 28), vmin=0, vmax=1.0, cmap='gray')
        ax.axis('off')
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-datadir', '--datadir')
    args = vars(parser.parse_args())
    data_found = False
    train_file = os.path.join(args['datadir'], 'Train-28x28_cntk_text.txt')
    if os.path.isfile(train_file):
        data_found = True
    if not data_found:
        raise ValueError('Please generate the data by completing CNTK 103 Part A')
    worker_rank = C.Communicator.rank()
    start = timer()
    reader_train = create_reader(train_file, True, d_input_dim, label_dim=10)
    (G_input, G_output, G_trainer_loss) = train(reader_train)
    C.Communicator.finalize()
    end = timer()
    print('Training loss of the generator at worker: {%d} is: {%f}, time taken is: {%d} seconds.' % (worker_rank, G_trainer_loss, end - start))