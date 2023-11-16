#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import argparse
import os

import tensorflow as tf
from bigdl.dllib.nncontext import init_nncontext
from bigdl.dllib.feature.dataset import mnist
from bigdl.orca.tfpark import KerasModel
from bigdl.dllib.utils.common import *
from bigdl.dllib.utils.log4Error import invalidInputError

parser = argparse.ArgumentParser(description="Run the tfpark keras "
                                             "dataset example.")
parser.add_argument('--max_epoch', type=int, default=5,
                    help='Set max_epoch for training, it should be integer.')
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The mode for the Spark cluster. local, yarn or spark-submit.')


def main(max_epoch):
    args = parser.parse_args()
    cluster_mode = args.cluster_mode
    if cluster_mode.startswith("yarn"):
        hadoop_conf = os.environ.get("HADOOP_CONF_DIR")
        invalidInputError(
            hadoop_conf is not None,
            "Directory path to hadoop conf not found for yarn-client mode. Please "
            "set the environment variable HADOOP_CONF_DIR")
        spark_conf = create_spark_conf().set("spark.executor.memory", "5g") \
            .set("spark.executor.cores", 2) \
            .set("spark.executor.instances", 2) \
            .set("spark.driver.memory", "2g")
        if cluster_mode == "yarn-client":
            _ = init_nncontext(spark_conf, cluster_mode="yarn-client", hadoop_conf=hadoop_conf)
        else:
            _ = init_nncontext(spark_conf, cluster_mode="yarn-cluster", hadoop_conf=hadoop_conf)
    else:
        _ = init_nncontext()

    (training_images_data, training_labels_data) = mnist.read_data_sets("/tmp/mnist", "train")
    (testing_images_data, testing_labels_data) = mnist.read_data_sets("/tmp/mnist", "test")

    training_images_data = (training_images_data - mnist.TRAIN_MEAN) / mnist.TRAIN_STD
    testing_images_data = (testing_images_data - mnist.TRAIN_MEAN) / mnist.TRAIN_STD

    model = tf.keras.Sequential(
        [tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
         tf.keras.layers.Dense(64, activation='relu'),
         tf.keras.layers.Dense(64, activation='relu'),
         tf.keras.layers.Dense(10, activation='softmax'),
         ]
    )

    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    keras_model = KerasModel(model)

    keras_model.fit(training_images_data,
                    training_labels_data,
                    validation_data=(testing_images_data, testing_labels_data),
                    epochs=max_epoch,
                    batch_size=320,
                    distributed=True)

    result = keras_model.evaluate(testing_images_data, testing_labels_data,
                                  distributed=True, batch_per_thread=80)

    print(result)
    # >> [0.08865142822265625, 0.9722]

    # the following is used for internal testing
    invalidInputError(result['acc Top1Accuracy'] > 0.95, "accuracy not reached 0.95")

    keras_model.save_weights("/tmp/mnist_keras.h5")


if __name__ == '__main__':

    args = parser.parse_args()
    max_epoch = args.max_epoch

    main(max_epoch)
