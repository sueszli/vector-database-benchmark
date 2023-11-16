import argparse
import logging
from typing import Iterable
from typing import Tuple
import numpy
import apache_beam as beam
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.tensorflow_inference import ModelType
from apache_beam.ml.inference.tensorflow_inference import TFModelHandlerNumpy
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.runners.runner import PipelineResult

def process_input(row: str) -> Tuple[int, numpy.ndarray]:
    if False:
        print('Hello World!')
    data = row.split(',')
    (label, pixels) = (int(data[0]), data[1:])
    pixels = [int(pixel) for pixel in pixels]
    pixels = numpy.array(pixels).reshape((28, 28, 1))
    return (label, pixels)

class PostProcessor(beam.DoFn):
    """Process the PredictionResult to get the predicted label.
  Returns a comma separated string with true label and predicted label.
  """

    def process(self, element: Tuple[int, PredictionResult]) -> Iterable[str]:
        if False:
            while True:
                i = 10
        (label, prediction_result) = element
        prediction = numpy.argmax(prediction_result.inference, axis=0)
        yield '{},{}'.format(label, prediction)

def parse_known_args(argv):
    if False:
        for i in range(10):
            print('nop')
    'Parses args for the workflow.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', required=True, help='text file with comma separated int values.')
    parser.add_argument('--output', dest='output', required=True, help='Path to save output predictions.')
    parser.add_argument('--model_path', dest='model_path', required=True, help='Path to load the Tensorflow model for Inference.')
    parser.add_argument('--large_model', action='store_true', dest='large_model', default=False, help='Set to true if your model is large enough to run into memory pressure if you load multiple copies.')
    return parser.parse_known_args(argv)

def run(argv=None, save_main_session=True, test_pipeline=None) -> PipelineResult:
    if False:
        for i in range(10):
            print('nop')
    '\n  Args:\n    argv: Command line arguments defined for this example.\n    save_main_session: Used for internal testing.\n    test_pipeline: Used for internal testing.\n  '
    (known_args, pipeline_args) = parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    model_loader = KeyedModelHandler(TFModelHandlerNumpy(model_uri=known_args.model_path, model_type=ModelType.SAVED_MODEL, large_model=known_args.large_model))
    pipeline = test_pipeline
    if not test_pipeline:
        pipeline = beam.Pipeline(options=pipeline_options)
    label_pixel_tuple = pipeline | 'ReadFromInput' >> beam.io.ReadFromText(known_args.input) | 'PreProcessInputs' >> beam.Map(process_input)
    predictions = label_pixel_tuple | 'RunInference' >> RunInference(model_loader) | 'PostProcessOutputs' >> beam.ParDo(PostProcessor())
    _ = predictions | 'WriteOutput' >> beam.io.WriteToText(known_args.output, shard_name_template='', append_trailing_newlines=True)
    result = pipeline.run()
    result.wait_until_finish()
    return result
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()