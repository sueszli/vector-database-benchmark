"""A pipeline that uses RunInference API to perform image classification."""
import argparse
import io
import logging
import os
from typing import Iterator
from typing import Optional
from typing import Tuple
import apache_beam as beam
import torch
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.pytorch_inference import PytorchModelHandlerTensor
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.runners.runner import PipelineResult
from PIL import Image
from torchvision import models
from torchvision import transforms

def read_image(image_file_name: str, path_to_dir: Optional[str]=None) -> Tuple[str, Image.Image]:
    if False:
        print('Hello World!')
    if path_to_dir is not None:
        image_file_name = os.path.join(path_to_dir, image_file_name)
    with FileSystems().open(image_file_name, 'r') as file:
        data = Image.open(io.BytesIO(file.read())).convert('RGB')
        return (image_file_name, data)

def preprocess_image(data: Image.Image) -> torch.Tensor:
    if False:
        return 10
    image_size = (224, 224)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), normalize])
    return transform(data)

def filter_empty_lines(text: str) -> Iterator[str]:
    if False:
        for i in range(10):
            print('nop')
    if len(text.strip()) > 0:
        yield text

def parse_known_args(argv):
    if False:
        return 10
    'Parses args for the workflow.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', required=True, help='Path to the text file containing image names.')
    parser.add_argument('--output', dest='output', required=True, help='Path where to save output predictions. text file.')
    parser.add_argument('--model_state_dict_path', dest='model_state_dict_path', required=True, help="Path to the model's state_dict.")
    parser.add_argument('--images_dir', default=None, help='Path to the directory where images are stored.Not required if image names in the input file have absolute path.')
    return parser.parse_known_args(argv)

def run(argv=None, model_class=None, model_params=None, save_main_session=True, device='CPU', test_pipeline=None) -> PipelineResult:
    if False:
        for i in range(10):
            print('nop')
    '\n  Args:\n    argv: Command line arguments defined for this example.\n    model_class: Reference to the class definition of the model.\n    model_params: Parameters passed to the constructor of the model_class.\n                  These will be used to instantiate the model object in the\n                  RunInference API.\n    save_main_session: Used for internal testing.\n    device: Device to be used on the Runner. Choices are (CPU, GPU).\n    test_pipeline: Used for internal testing.\n  '
    (known_args, pipeline_args) = parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    if not model_class:
        model_class = models.mobilenet_v2
        model_params = {'num_classes': 1000}

    def preprocess(image_name: str) -> Tuple[str, torch.Tensor]:
        if False:
            i = 10
            return i + 15
        (image_name, image) = read_image(image_file_name=image_name, path_to_dir=known_args.images_dir)
        return (image_name, preprocess_image(image))

    def postprocess(element: Tuple[str, PredictionResult]) -> str:
        if False:
            return 10
        (filename, prediction_result) = element
        prediction = torch.argmax(prediction_result.inference, dim=0)
        return filename + ',' + str(prediction.item())
    model_handler = KeyedModelHandler(PytorchModelHandlerTensor(state_dict_path=known_args.model_state_dict_path, model_class=model_class, model_params=model_params, device=device, min_batch_size=10, max_batch_size=100)).with_preprocess_fn(preprocess).with_postprocess_fn(postprocess)
    pipeline = test_pipeline
    if not test_pipeline:
        pipeline = beam.Pipeline(options=pipeline_options)
    filename_value_pair = pipeline | 'ReadImageNames' >> beam.io.ReadFromText(known_args.input) | 'FilterEmptyLines' >> beam.ParDo(filter_empty_lines)
    predictions = filename_value_pair | 'PyTorchRunInference' >> RunInference(model_handler)
    predictions | 'WriteOutputToGCS' >> beam.io.WriteToText(known_args.output, shard_name_template='', append_trailing_newlines=True)
    result = pipeline.run()
    result.wait_until_finish()
    return result
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()