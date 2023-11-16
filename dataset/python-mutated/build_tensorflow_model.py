from typing import Dict
from typing import Optional
import tensorflow as tf

class TFModelWrapperWithSignature(tf.keras.Model):
    """
  Helper class used to wrap a based tf.keras.Model object with a serving
  signature that can passed to the tfx_bsl RunInference transform.

  A TF model saved using this helper class expects inputs as
    images serialized to tf.string using tf.io.parse_tensor
    and then passing serialized images to the RunInference transform
    in the tf.train.Example. More about tf.train.Example at
    https://www.tensorflow.org/api_docs/python/tf/train/Example

  Usage:
  Step 1:
  # Save the base TF model with modified signature .
  signature_model = TFModelWrapperWithSignature(
      model=model,
      preprocess_input=preprocess_input,
      input_dtype=input_dtype,
      feature_description=feature_description,
      **kwargs
      )
  tf.saved_model.save(signature_model, path)

  Step 2:
  # Load the saved_model in the beam pipeline to create ModelHandler.
  saved_model_spec = model_spec_pb2.SavedModelSpec(
      model_path=known_args.model_path)
  inferece_spec_type = model_spec_pb2.InferenceSpecType(
      saved_model_spec=saved_model_spec)
  model_handler = CreateModelHandler(inferece_spec_type)
  """

    def __init__(self, model, preprocess_input=None, input_dtype=None, feature_description=None, **kwargs):
        if False:
            while True:
                i = 10
        "\n    model: model: Base tensorflow model used for TFX-BSL RunInference transform.\n    preprocess_input: Preprocess method to be included as part of the\n      model's serving signature.\n    input_dtype: tf dtype of the inputs passed to the model.\n      For eg: tf.int32, tf.uint8.\n    feature_description: Feature spec to parse inputs from tf.train.Example\n      using tf.parse_example(). For more details, please take a look at\n      https://www.tensorflow.org/api_docs/python/tf/io/parse_example\n    If there are extra arguments(for eg: training=False) that should be\n    passed to the base tf model during inference, please pass them in kwargs.\n    "
        super().__init__()
        self.model = model
        self.preprocess_input = preprocess_input
        self.input_dtype = input_dtype
        self.feature_description = feature_description
        if not feature_description:
            self.feature_description = {'image': tf.io.FixedLenFeature((), tf.string)}
        self._kwargs = kwargs

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def call(self, serialized_examples):
        if False:
            for i in range(10):
                print('nop')
        features = tf.io.parse_example(serialized_examples, features=self.feature_description)
        num_batches = len(features['image'])
        deserialized_vectors = tf.TensorArray(self.input_dtype, size=num_batches, dynamic_size=True)
        for i in range(num_batches):
            deserialized_value = tf.io.parse_tensor(features['image'][i], out_type=self.input_dtype)
            deserialized_vectors = deserialized_vectors.write(i, deserialized_value)
        deserialized_tensor = deserialized_vectors.stack()
        if self.preprocess_input:
            deserialized_tensor = self.preprocess_input(deserialized_tensor)
        return self.model(deserialized_tensor, **self._kwargs)

def save_tf_model_with_signature(path_to_save_model, model=None, preprocess_input=None, input_dtype=tf.float32, feature_description: Optional[Dict]=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n  Helper function used to save the Tensorflow Model with a serving signature.\n  This is intended only for internal testing.\n\n  Args:\n   path_to_save_model: Path to save the model with modified signature.\n  model: model: Base tensorflow model used for TFX-BSL RunInference transform.\n  preprocess_input: Preprocess method to be included as part of the\n    model's serving signature.\n  input_dtype: tf dtype of the inputs passed to the model.\n    For eg: tf.int32, tf.uint8.\n  feature_description: Feature spec to parse inputs from tf.train.Example using\n    tf.parse_example(). For more details, please take a look at\n    https://www.tensorflow.org/api_docs/python/tf/io/parse_example\n\n  If there are extra arguments(for eg: training=False) that should be passed to\n  the base tf model during inference, please pass them in kwargs.\n  "
    if not model:
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    signature_model = TFModelWrapperWithSignature(model=model, preprocess_input=preprocess_input, input_dtype=input_dtype, feature_description=feature_description, **kwargs)
    tf.saved_model.save(signature_model, path_to_save_model)