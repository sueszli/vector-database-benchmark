"""This trains a TensorFlow Keras model to classify land cover.

The model is a simple Fully Convolutional Network (FCN).
"""
from __future__ import annotations
import tensorflow as tf
EPOCHS = 100
BATCH_SIZE = 512
KERNEL_SIZE = 5
NUM_INPUTS = 13
NUM_CLASSES = 9
TRAIN_TEST_RATIO = 90
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 8

def read_example(serialized: bytes) -> tuple[tf.Tensor, tf.Tensor]:
    if False:
        return 10
    'Parses and reads a training example from TFRecords.\n\n    Args:\n        serialized: Serialized example bytes from TFRecord files.\n\n    Returns: An (inputs, labels) pair of tensors.\n    '
    features_dict = {'inputs': tf.io.FixedLenFeature([], tf.string), 'labels': tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(serialized, features_dict)
    inputs = tf.io.parse_tensor(example['inputs'], tf.float32)
    labels = tf.io.parse_tensor(example['labels'], tf.uint8)
    inputs.set_shape([None, None, NUM_INPUTS])
    labels.set_shape([None, None, 1])
    one_hot_labels = tf.one_hot(labels[:, :, 0], NUM_CLASSES)
    return (inputs, one_hot_labels)

def read_dataset(data_path: str) -> tf.data.Dataset:
    if False:
        print('Hello World!')
    'Reads compressed TFRecord files from a directory into a tf.data.Dataset.\n\n    Args:\n        data_path: Local or Cloud Storage directory path where the TFRecord files are.\n\n    Returns: A tf.data.Dataset with the contents of the TFRecord files.\n    '
    file_pattern = tf.io.gfile.join(data_path, '*.tfrecord.gz')
    file_names = tf.data.Dataset.list_files(file_pattern).cache()
    dataset = tf.data.TFRecordDataset(file_names, compression_type='GZIP')
    return dataset.map(read_example, num_parallel_calls=tf.data.AUTOTUNE)

def split_dataset(dataset: tf.data.Dataset, batch_size: int=BATCH_SIZE, train_test_ratio: int=TRAIN_TEST_RATIO) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    if False:
        while True:
            i = 10
    'Splits a dataset into training and validation subsets.\n\n    Args:\n        dataset: Full dataset with all the training examples.\n        batch_size: Number of examples per training batch.\n        train_test_ratio: Percent of the data to use for training.\n\n    Returns: A (training, validation) dataset pair.\n    '
    indexed_dataset = dataset.enumerate()
    train_dataset = indexed_dataset.filter(lambda i, _: i % 100 <= train_test_ratio).map(lambda _, data: data, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset = indexed_dataset.filter(lambda i, _: i % 100 > train_test_ratio).map(lambda _, data: data, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return (train_dataset, validation_dataset)

def create_model(dataset: tf.data.Dataset, kernel_size: int=KERNEL_SIZE) -> tf.keras.Model:
    if False:
        return 10
    'Creates a Fully Convolutional Network Keras model.\n\n    Make sure you pass the *training* dataset, not the validation or full dataset.\n\n    Args:\n        dataset: Training dataset used to normalize inputs.\n        kernel_size: Size of the square of neighboring pixels for the model to look at.\n\n    Returns: A compiled fresh new model (not trained).\n    '
    normalization = tf.keras.layers.Normalization()
    normalization.adapt(dataset.map(lambda inputs, _: inputs))
    model = tf.keras.Sequential([tf.keras.Input(shape=(None, None, NUM_INPUTS)), normalization, tf.keras.layers.Conv2D(32, kernel_size, activation='relu'), tf.keras.layers.Conv2DTranspose(16, kernel_size, activation='relu'), tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.OneHotIoU(num_classes=NUM_CLASSES, target_class_ids=list(range(NUM_CLASSES)))])
    return model

def run(data_path: str, model_path: str, epochs: int=EPOCHS, batch_size: int=BATCH_SIZE, kernel_size: int=KERNEL_SIZE, train_test_ratio: int=TRAIN_TEST_RATIO) -> tf.keras.Model:
    if False:
        return 10
    'Creates and trains the model.\n\n    Args:\n        data_path: Local or Cloud Storage directory path where the TFRecord files are.\n        model_path: Local or Cloud Storage directory path to store the trained model.\n        epochs: Number of times the model goes through the training dataset during training.\n        batch_size: Number of examples per training batch.\n        kernel_size: Size of the square of neighboring pixels for the model to look at.\n        train_test_ratio: Percent of the data to use for training.\n\n    Returns: The trained model.\n    '
    print(f'data_path: {data_path}')
    print(f'model_path: {model_path}')
    print(f'epochs: {epochs}')
    print(f'batch_size: {batch_size}')
    print(f'kernel_size: {kernel_size}')
    print(f'train_test_ratio: {train_test_ratio}')
    print('-' * 40)
    dataset = read_dataset(data_path)
    (train_dataset, test_dataset) = split_dataset(dataset, batch_size, train_test_ratio)
    model = create_model(train_dataset, kernel_size)
    print(model.summary())
    model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)
    model.save(model_path)
    print(f'Model saved to path: {model_path}')
    return model
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='Local or Cloud Storage directory path where the TFRecord files are.')
    parser.add_argument('--model-path', required=True, help='Local or Cloud Storage directory path to store the trained model.')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of times the model goes through the training dataset during training.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Number of examples per training batch.')
    parser.add_argument('--kernel-size', type=int, default=KERNEL_SIZE, help='Size of the square of neighboring pixels for the model to look at.')
    parser.add_argument('--train-test-ratio', type=int, default=TRAIN_TEST_RATIO, help='Percent of the data to use for training.')
    args = parser.parse_args()
    run(data_path=args.data_path, model_path=args.model_path, epochs=args.epochs, batch_size=args.batch_size, kernel_size=args.kernel_size, train_test_ratio=args.train_test_ratio)