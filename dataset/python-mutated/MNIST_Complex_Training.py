from __future__ import print_function
import os
import cntk as C
import numpy as np
import scipy.sparse
input_shape = (28, 28)
num_classes = 10
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Models/mnist.cmf')
try:
    from sklearn import datasets, utils
    mnist = datasets.fetch_mldata('MNIST original')
    (X, Y) = (mnist.data / 255.0, mnist.target)
    (X_train, X_test) = (X[:60000].reshape((-1, 28, 28)), X[60000:].reshape((-1, 28, 28)))
    (Y_train, Y_test) = (Y[:60000].astype(int), Y[60000:].astype(int))
except:
    import requests, io, gzip
    (X_train, X_test) = (np.fromstring(gzip.GzipFile(fileobj=io.BytesIO(requests.get('http://yann.lecun.com/exdb/mnist/' + name + '-images-idx3-ubyte.gz').content)).read()[16:], dtype=np.uint8).reshape((-1, 28, 28)).astype(np.float32) / 255.0 for name in ('train', 't10k'))
    (Y_train, Y_test) = (np.fromstring(gzip.GzipFile(fileobj=io.BytesIO(requests.get('http://yann.lecun.com/exdb/mnist/' + name + '-labels-idx1-ubyte.gz').content)).read()[8:], dtype=np.uint8).astype(int) for name in ('train', 't10k'))
np.random.seed(0)
idx = np.random.permutation(len(X_train))
(X_train, Y_train) = (X_train[idx], Y_train[idx])
(X_train, X_cv) = (X_train[:54000], X_train[54000:])
(Y_train, Y_cv) = (Y_train[:54000], Y_train[54000:])
(Y_train, Y_cv, Y_test) = (scipy.sparse.csr_matrix((np.ones(len(Y), np.float32), (range(len(Y)), Y)), shape=(len(Y), 10)) for Y in (Y_train, Y_cv, Y_test))
(X_train, X_cv, X_test) = (X.astype(np.float32) for X in (X_train, X_cv, X_test))
with C.layers.default_options(activation=C.ops.relu, pad=False):
    model = C.layers.Sequential([C.layers.Convolution2D((5, 5), num_filters=32, reduction_rank=0, pad=True), C.layers.MaxPooling((2, 2), strides=(2, 2)), C.layers.Convolution2D((3, 3), num_filters=48), C.layers.MaxPooling((2, 2), strides=(2, 2)), C.layers.Convolution2D((3, 3), num_filters=64), C.layers.Dense(96), C.layers.Dropout(dropout_rate=0.5), C.layers.Dense(num_classes, activation=None)])

@C.Function.with_signature(C.layers.Tensor[input_shape], C.layers.SparseTensor[num_classes])
def criterion(data, label_one_hot):
    if False:
        i = 10
        return i + 15
    z = model(data)
    loss = C.cross_entropy_with_softmax(z, label_one_hot)
    metric = C.classification_error(z, label_one_hot)
    return (loss, metric)
epoch_size = len(X_train)
lr_per_sample = 0.001
lr_schedule = C.learning_parameter_schedule_per_sample(lr_per_sample)
mm_per_sample = [0] * 5 + [0.9990239141819757]
mm_schedule = C.learners.momentum_schedule_per_sample(mm_per_sample, epoch_size=epoch_size)
learner = C.learners.momentum_sgd(model.parameters, lr_schedule, mm_schedule)
progress_writer = C.logging.ProgressPrinter()
checkpoint_callback_config = C.CheckpointConfig(model_path, epoch_size, restore=False)
prev_metric = 1

def adjust_lr_callback(index, average_error, cv_num_samples, cv_num_minibatches):
    if False:
        print('Hello World!')
    global prev_metric
    if (prev_metric - average_error) / prev_metric < 0.05:
        learner.reset_learning_rate(C.learning_parameter_schedule_per_sample(learner.learning_rate() / 2))
        if learner.learning_rate() < lr_per_sample / (2 ** 7 - 0.1):
            print('Learning rate {} too small. Training complete.'.format(learner.learning_rate()))
            return False
        print('Improvement of metric from {:.3f} to {:.3f} insufficient. Halving learning rate to {}.'.format(prev_metric, average_error, learner.learning_rate()))
    prev_metric = average_error
    return True
cv_callback_config = C.CrossValidationConfig((X_cv, Y_cv), 3 * epoch_size, minibatch_size=256, callback=adjust_lr_callback, criterion=criterion)
test_callback_config = C.TestConfig((X_test, Y_test), criterion=criterion)
learner = C.train.distributed.data_parallel_distributed_learner(learner)
minibatch_size_schedule = C.minibatch_size_schedule([256] * 6 + [512] * 9 + [1024] * 7 + [2048] * 8 + [4096], epoch_size=epoch_size)
progress = criterion.train((X_train, Y_train), minibatch_size=minibatch_size_schedule, max_epochs=50, parameter_learners=[learner], callbacks=[progress_writer, checkpoint_callback_config, cv_callback_config, test_callback_config])
final_loss = progress.epoch_summaries[-1].loss
final_metric = progress.epoch_summaries[-1].metric
final_samples = progress.epoch_summaries[-1].samples
test_metric = progress.test_summary.metric

@C.Function.with_signature(C.layers.Tensor[input_shape])
def get_probability(data):
    if False:
        while True:
            i = 10
    return C.softmax(model(data))
(X_check, Y_check) = (X_test[0:10000:400].copy(), Y_test[0:10000:400])
result = get_probability(X_check)
print('Label    :', [label.todense().argmax() for label in Y_check])
print('Predicted:', [result[i, :].argmax() for i in range(len(result))])
C.train.distributed.Communicator.finalize()