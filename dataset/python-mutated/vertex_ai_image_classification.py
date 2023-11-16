""" A sample pipeline using the RunInference API to classify images of flowers.
This pipeline reads an already-processes representation of an image of
sunflowers and sends it to a deployed Vertex AI model endpoint, then
returns the predictions from the classifier model. The model and image
are from the Hello Image Data Vertex AI tutorial (see
https://cloud.google.com/vertex-ai/docs/tutorials/image-recognition-custom
for more information.)
"""
import argparse
import io
import logging
from typing import Iterable
from typing import List
from typing import Tuple
import apache_beam as beam
import tensorflow as tf
from apache_beam.io import fileio
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.vertex_ai_inference import VertexAIModelHandlerJSON
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.runners.runner import PipelineResult

def parse_known_args(argv):
    if False:
        print('Hello World!')
    'Parses args for the workflow.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', type=str, required=True, help='File glob to read images from.')
    parser.add_argument('--output', dest='output', type=str, required=True, help='Path to save output predictions.')
    parser.add_argument('--endpoint_id', dest='endpoint', type=str, required=True, help='Vertex AI Endpoint resource ID to query (string).')
    parser.add_argument('--endpoint_project', dest='project', required=True, help='GCP Project')
    parser.add_argument('--endpoint_region', dest='location', type=str, required=True, help='GCP location for the Endpoint')
    parser.add_argument('--endpoint_network', dest='vpc_network', type=str, required=False, help='GCP network the endpoint is peered to')
    parser.add_argument('--experiment', dest='experiment', type=str, required=False, help='Vertex AI experiment label to apply to queries')
    parser.add_argument('--private', dest='private', type=bool, default=False, help='True if the Vertex AI endpoint is a private endpoint')
    return parser.parse_known_args(argv)
IMG_WIDTH = 128
COLUMNS = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

def read_image(image_file_name: str) -> Tuple[str, bytes]:
    if False:
        i = 10
        return i + 15
    with FileSystems().open(image_file_name, 'r') as file:
        data = io.BytesIO(file.read()).getvalue()
        return (image_file_name, data)

def preprocess_image(data: bytes) -> List[float]:
    if False:
        while True:
            i = 10
    'Preprocess the image, resizing it and normalizing it before\n  converting to a list.\n  '
    image = tf.io.decode_jpeg(data, channels=3)
    image = tf.image.resize_with_pad(image, IMG_WIDTH, IMG_WIDTH)
    image = image / 255
    return image.numpy().tolist()

class PostProcessor(beam.DoFn):

    def process(self, element: Tuple[str, PredictionResult]) -> Iterable[str]:
        if False:
            return 10
        (img_name, prediction_result) = element
        prediction_vals = prediction_result.inference
        index = prediction_vals.index(max(prediction_vals))
        yield (img_name + ': ' + str(COLUMNS[index]) + ' (' + str(max(prediction_vals)) + ')')

def run(argv=None, save_main_session=True, test_pipeline=None) -> PipelineResult:
    if False:
        print('Hello World!')
    '\n  Args:\n    argv: Command line arguments defined for this example.\n    save_main_session: Used for internal testing.\n    test_pipeline: Used for internal testing.\n  '
    (known_args, pipeline_args) = parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    model_handler = VertexAIModelHandlerJSON(endpoint_id=known_args.endpoint, project=known_args.project, location=known_args.location, experiment=known_args.experiment, network=known_args.vpc_network, private=known_args.private)
    pipeline = test_pipeline
    if not test_pipeline:
        pipeline = beam.Pipeline(options=pipeline_options)
    read_glob = pipeline | 'Get glob' >> beam.Create([known_args.input])
    read_image_name = read_glob | 'Get Image Paths' >> fileio.MatchAll()
    load_image = read_image_name | 'Read Image' >> beam.Map(lambda image_name: read_image(image_name.path))
    preprocess = load_image | 'Preprocess Image' >> beam.MapTuple(lambda img_name, img: (img_name, preprocess_image(img)))
    predictions = preprocess | 'RunInference' >> RunInference(KeyedModelHandler(model_handler))
    process_output = predictions | 'Process Predictions' >> beam.ParDo(PostProcessor())
    _ = process_output | 'WriteOutput' >> beam.io.WriteToText(known_args.output, shard_name_template='', append_trailing_newlines=True)
    result = pipeline.run()
    result.wait_until_finish()
    return result
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()