from __future__ import absolute_import, division, print_function
import os
import cv2
import numpy as np
import tensorflow as tf
import hipCNN

def main():
    if False:
        for i in range(10):
            print('nop')
    print('start')
    ((train_data, train_labels), (eval_data, eval_labels)) = load_images('../data/finish-line/bmps/train/')
    train_data = train_data / np.float32(255)
    train_labels = train_labels.astype(np.int32)
    eval_data = eval_data / np.float32(255)
    eval_labels = eval_labels.astype(np.int32)
    hip_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir='model')
    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': train_data}, y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
    hip_classifier.train(input_fn=train_input_fn, steps=1000, hooks=[logging_hook])
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = hip_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
main()