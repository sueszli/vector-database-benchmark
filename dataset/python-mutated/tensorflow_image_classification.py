"""
A sample pipeline illustrating how to use Apache Beam RunInference
with TFX_BSL CreateModelHandler API. For more details, please look at
https://www.tensorflow.org/tfx/tfx_bsl/api_docs/python/tfx_bsl/public/beam/run_inference/CreateModelHandler.

Note: A Tensorflow Model needs to be updated with a @tf.function
      signature in order to accept bytes as inputs, and should have logic
      to decode bytes to Tensors that is acceptable by the TensorFlow model.
      Please take a look at TFModelWrapperWithSignature class in
      build_tensorflow_model.py on how to modify TF Model's signature
      and the logic to decode the image tensor.
"""
import argparse
import io
import logging
import os
from typing import Iterable
from typing import Iterator
from typing import Optional
from typing import Tuple
import apache_beam as beam
import tensorflow as tf
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.base import RunInference
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.runners.runner import PipelineResult
from PIL import Image
from tfx_bsl.public.beam.run_inference import CreateModelHandler
from tfx_bsl.public.beam.run_inference import prediction_log_pb2
from tfx_bsl.public.proto import model_spec_pb2
_IMG_SIZE = (224, 224)

def filter_empty_lines(text: str) -> Iterator[str]:
    if False:
        print('Hello World!')
    if len(text.strip()) > 0:
        yield text

def read_and_process_image(image_file_name: str, path_to_dir: Optional[str]=None) -> Tuple[str, tf.Tensor]:
    if False:
        for i in range(10):
            print('nop')
    if path_to_dir is not None:
        image_file_name = os.path.join(path_to_dir, image_file_name)
    with FileSystems().open(image_file_name, 'r') as file:
        data = Image.open(io.BytesIO(file.read())).convert('RGB')
    image = tf.keras.preprocessing.image.img_to_array(data)
    image = tf.image.resize(image, _IMG_SIZE)
    return (image_file_name, image)

def convert_image_to_example_proto(tensor: tf.Tensor) -> tf.train.Example:
    if False:
        while True:
            i = 10
    '\n  This method performs the following:\n  1. Accepts the tensor as input\n  2. Serializes the tensor into bytes and pass it through\n        tf.train.Feature\n  3. Pass the serialized tensor feature using tf.train.Example\n      Proto to the RunInference transform.\n\n  Args:\n    tensor: A TF tensor.\n  Returns:\n    example_proto: A tf.train.Example containing serialized tensor.\n  '
    serialized_non_scalar = tf.io.serialize_tensor(tensor)
    feature_of_bytes = tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_non_scalar.numpy()]))
    features_for_example = {'image': feature_of_bytes}
    example_proto = tf.train.Example(features=tf.train.Features(feature=features_for_example))
    return example_proto

class ProcessInferenceToString(beam.DoFn):

    def process(self, element: Tuple[str, prediction_log_pb2.PredictionLog]) -> Iterable[str]:
        if False:
            return 10
        '\n    Args:\n      element: Tuple of str, and PredictionLog. Inference can be parsed\n        from prediction_log\n    returns:\n      str of filename and inference.\n    '
        (filename, predict_log) = (element[0], element[1].predict_log)
        output_value = predict_log.response.outputs
        output_tensor = tf.io.decode_raw(output_value['output_0'].tensor_content, out_type=tf.float32)
        max_index_output_tensor = tf.math.argmax(output_tensor, axis=0)
        yield (filename + ',' + str(tf.get_static_value(max_index_output_tensor)))

def parse_known_args(argv):
    if False:
        i = 10
        return i + 15
    'Parses args for the workflow.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', required=True, help='Path to the text file containing image names.')
    parser.add_argument('--output', dest='output', required=True, help='Path to save output predictions text file.')
    parser.add_argument('--model_path', dest='model_path', required=True, help='Path to the model.')
    parser.add_argument('--images_dir', default=None, help='Path to the directory where images are stored.Not required if image names in the input file have absolute path.')
    return parser.parse_known_args(argv)

def run(argv=None, save_main_session=True, pipeline=None) -> PipelineResult:
    if False:
        for i in range(10):
            print('nop')
    '\n  Args:\n    argv: Command line arguments defined for this example.\n    save_main_session: Used for internal testing.\n    test_pipeline: Used for internal testing.\n  '
    (known_args, pipeline_args) = parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    saved_model_spec = model_spec_pb2.SavedModelSpec(model_path=known_args.model_path)
    inferece_spec_type = model_spec_pb2.InferenceSpecType(saved_model_spec=saved_model_spec)
    model_handler = CreateModelHandler(inferece_spec_type)
    keyed_model_handler = KeyedModelHandler(model_handler)
    if not pipeline:
        pipeline = beam.Pipeline(options=pipeline_options)
    filename_value_pair = pipeline | 'ReadImageNames' >> beam.io.ReadFromText(known_args.input) | 'FilterEmptyLines' >> beam.ParDo(filter_empty_lines) | 'ProcessImageData' >> beam.Map(lambda image_name: read_and_process_image(image_file_name=image_name, path_to_dir=known_args.images_dir))
    predictions = filename_value_pair | 'ConvertToExampleProto' >> beam.Map(lambda x: (x[0], convert_image_to_example_proto(x[1]))) | 'TFXRunInference' >> RunInference(keyed_model_handler) | 'PostProcess' >> beam.ParDo(ProcessInferenceToString())
    _ = predictions | 'WriteOutputToGCS' >> beam.io.WriteToText(known_args.output, shard_name_template='', append_trailing_newlines=True)
    result = pipeline.run()
    result.wait_until_finish()
    return result
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()