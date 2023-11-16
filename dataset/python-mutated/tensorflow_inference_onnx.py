import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import ResNet50
from bigdl.nano.tf.keras import InferenceOptimizer

def create_dataset(img_size, batch_size):
    if False:
        for i in range(10):
            print('nop')
    (dataset, info) = tfds.load('imagenette/320px-v2', data_dir='/tmp/data', split='validation[:5%]', with_info=True, as_supervised=True)
    num_classes = info.features['label'].num_classes

    def preprocessing(img, label):
        if False:
            return 10
        return (tf.image.resize(img, (img_size, img_size)), tf.one_hot(label, num_classes))
    dataset = dataset.map(preprocessing).batch(batch_size)
    return dataset
if __name__ == '__main__':
    img_size = 224
    batch_size = 32
    dataset = create_dataset(img_size, batch_size)
    model = ResNet50(weights='imagenet', input_shape=(img_size, img_size, 3))
    preds = model.predict(dataset)
    spec = tf.TensorSpec((None, 224, 224, 3), tf.float32)
    onnx_model = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_spec=spec)
    onnx_preds = onnx_model.predict(dataset)
    np.testing.assert_allclose(preds, onnx_preds, rtol=0.0001)