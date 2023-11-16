from __future__ import print_function
import cntk
import numpy as np
import scipy.sparse
input_dim = 2
num_classes = 2
np.random.seed(0)

def generate_synthetic_data(N):
    if False:
        return 10
    Y = np.random.randint(size=N, low=0, high=num_classes)
    X = (np.random.randn(N, input_dim) + 3) * (Y[:, None] + 1)
    Y = scipy.sparse.csr_matrix((np.ones(N, np.float32), (range(N), Y)), shape=(N, num_classes))
    X = X.astype(np.float32)
    return (X, Y)
(X_train, Y_train) = generate_synthetic_data(20000)
(X_test, Y_test) = generate_synthetic_data(1024)
data = cntk.input_variable(input_dim)
W = cntk.Parameter((input_dim, num_classes), init=cntk.glorot_uniform(), name='W')
b = cntk.Parameter((num_classes,), init=0, name='b')
model = cntk.times(data, W) + b
label_one_hot = cntk.input_variable(num_classes, is_sparse=True)
loss = cntk.cross_entropy_with_softmax(model, label_one_hot)
metric = cntk.classification_error(model, label_one_hot)
criterion = cntk.combine([loss, metric])
learning_rate = 0.1
learner = cntk.sgd(model.parameters, cntk.learning_parameter_schedule(learning_rate))
minibatch_size = 32
progress_writer = cntk.logging.ProgressPrinter(50)
trainer = cntk.Trainer(None, criterion, [learner], [progress_writer])
for i in range(0, len(X_train), minibatch_size):
    x = X_train[i:i + minibatch_size]
    y = Y_train[i:i + minibatch_size]
    trainer.train_minibatch({data: x, label_one_hot: y})
trainer.summarize_training_progress()
evaluator = cntk.Evaluator(metric, [progress_writer])
for i in range(0, len(X_test), minibatch_size):
    x = X_test[i:i + minibatch_size]
    y = Y_test[i:i + minibatch_size]
    evaluator.test_minibatch({data: x, label_one_hot: y})
evaluator.summarize_test_progress()
get_probability = cntk.softmax(model)
(X_check, Y_check) = generate_synthetic_data(25)
result = get_probability.eval(X_check)
print('Label    :', [label.todense().argmax() for label in Y_check])
print('Predicted:', [result[i, :].argmax() for i in range(len(result))])