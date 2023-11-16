from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import mnistCNN

def main():
    if False:
        return 10
    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
    train_data = train_data / np.float32(255)
    train_labels = train_labels.astype(np.int32)
    eval_data = eval_data / np.float32(255)
    eval_labels = eval_labels.astype(np.int32)
    mnist_classifier = tf.estimator.Estimator(model_fn=mnistCNN.cnn_model_fn, model_dir='mnist')
    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': train_data}, y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn, steps=2000)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
main()