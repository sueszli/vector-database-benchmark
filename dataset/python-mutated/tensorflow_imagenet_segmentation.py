import argparse
import logging
from typing import Iterable
from typing import Iterator
import numpy
import apache_beam as beam
import tensorflow as tf
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.tensorflow_inference import TFModelHandlerTensor
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.runners.runner import PipelineResult
from PIL import Image

class PostProcessor(beam.DoFn):
    """Process the PredictionResult to get the predicted label.
  Returns predicted label.
  """

    def setup(self):
        if False:
            return 10
        labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
        self._imagenet_labels = numpy.array(open(labels_path).read().splitlines())

    def process(self, element: PredictionResult) -> Iterable[str]:
        if False:
            return 10
        predicted_class = numpy.argmax(element.inference, axis=-1)
        predicted_class_name = self._imagenet_labels[predicted_class]
        yield predicted_class_name.title()

def parse_known_args(argv):
    if False:
        return 10
    'Parses args for the workflow.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', required=True, help='Path to the text file containing image names.')
    parser.add_argument('--output', dest='output', required=True, help='Path to save output predictions.')
    parser.add_argument('--model_path', dest='model_path', required=True, help='Path to load the Tensorflow model for Inference.')
    parser.add_argument('--image_dir', help='Path to the directory where images are stored.')
    return parser.parse_known_args(argv)

def filter_empty_lines(text: str) -> Iterator[str]:
    if False:
        i = 10
        return i + 15
    if len(text.strip()) > 0:
        yield text

def read_image(image_name, image_dir):
    if False:
        return 10
    img = tf.keras.utils.get_file(image_name, image_dir + image_name)
    img = Image.open(img).resize((224, 224))
    img = numpy.array(img) / 255.0
    img_tensor = tf.cast(tf.convert_to_tensor(img[...]), dtype=tf.float32)
    return img_tensor

def run(argv=None, save_main_session=True, test_pipeline=None) -> PipelineResult:
    if False:
        print('Hello World!')
    '\n  Args:\n    argv: Command line arguments defined for this example.\n    save_main_session: Used for internal testing.\n    test_pipeline: Used for internal testing.\n  '
    (known_args, pipeline_args) = parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    model_loader = TFModelHandlerTensor(model_uri=known_args.model_path).with_preprocess_fn(lambda image_name: read_image(image_name, known_args.image_dir))
    pipeline = test_pipeline
    if not test_pipeline:
        pipeline = beam.Pipeline(options=pipeline_options)
    image = pipeline | 'ReadImageNames' >> beam.io.ReadFromText(known_args.input) | 'FilterEmptyLines' >> beam.ParDo(filter_empty_lines)
    predictions = image | 'RunInference' >> RunInference(model_loader) | 'PostProcessOutputs' >> beam.ParDo(PostProcessor())
    _ = predictions | 'WriteOutput' >> beam.io.WriteToText(known_args.output, shard_name_template='', append_trailing_newlines=True)
    result = pipeline.run()
    result.wait_until_finish()
    return result
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()