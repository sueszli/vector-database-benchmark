from __future__ import print_function
import cntk
import numpy as np
import scipy.sparse
input_dim = 2
num_classes = 2
np.random.seed(0)

def generate_synthetic_data(N):
    if False:
        while True:
            i = 10
    Y = np.random.randint(size=N, low=0, high=num_classes)
    X = (np.random.randn(N, input_dim) + 3) * (Y[:, None] + 1)
    Y = scipy.sparse.csr_matrix((np.ones(N, np.float32), (range(N), Y)), shape=(N, num_classes))
    X = X.astype(np.float32)
    return (X, Y)
(X_train, Y_train) = generate_synthetic_data(20000)
(X_test, Y_test) = generate_synthetic_data(1024)
model = cntk.layers.Dense(num_classes, activation=None)

@cntk.Function.with_signature(cntk.layers.Tensor[input_dim], cntk.layers.SparseTensor[num_classes])
def criterion(data, label_one_hot):
    if False:
        i = 10
        return i + 15
    z = model(data)
    loss = cntk.cross_entropy_with_softmax(z, label_one_hot)
    metric = cntk.classification_error(z, label_one_hot)
    return (loss, metric)
learning_rate = 0.1
learner = cntk.sgd(model.parameters, cntk.learning_parameter_schedule(learning_rate))
progress_writer = cntk.logging.ProgressPrinter(50)
progress = criterion.train((X_train, Y_train), parameter_learners=[learner], callbacks=[progress_writer])
(final_loss, final_metric, final_samples) = (progress.epoch_summaries[-1].loss, progress.epoch_summaries[-1].metric, progress.epoch_summaries[-1].samples)
test_metric = criterion.test((X_test, Y_test), callbacks=[progress_writer]).metric

@cntk.Function.with_signature(cntk.layers.Tensor[input_dim])
def get_probability(data):
    if False:
        i = 10
        return i + 15
    return cntk.softmax(model(data))
(X_check, Y_check) = generate_synthetic_data(25)
result = get_probability(X_check)
print('Label    :', [label.todense().argmax() for label in Y_check])
print('Predicted:', [result[i, :].argmax() for i in range(len(result))])