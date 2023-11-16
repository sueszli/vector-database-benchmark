from __future__ import absolute_import, division, print_function
import os
import cv2
import numpy as np
import tensorflow as tf
import hipCNN

# This function runs 1000 steps of training for the
# hip recognizing CNN. The CNN is trained in batch 
# sizes of 100 which means this runs about 10 epochs
def main():
    print("start")
    ((train_data, train_labels),
    (eval_data, eval_labels)) = load_images("../data/finish-line/bmps/train/")



    train_data = train_data/np.float32(255)
    train_labels = train_labels.astype(np.int32)  # not required

    eval_data = eval_data/np.float32(255)
    eval_labels = eval_labels.astype(np.int32)  # not required


    # Create the Estimator
    hip_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="model")


    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

    hip_classifier.train(
    input_fn=train_input_fn,
    steps=1000,
    hooks=[logging_hook])


    # hip_classifier.train(input_fn=train_input_fn, steps=100)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

    eval_results = hip_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

main()
