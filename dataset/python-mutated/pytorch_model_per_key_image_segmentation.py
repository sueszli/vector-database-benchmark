"""
A pipeline that uses RunInference API to perform image segmentation using
multiple different models.
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
import torch
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.base import KeyModelMapping
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.pytorch_inference import PytorchModelHandlerTensor
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.runners.runner import PipelineResult
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
COCO_INSTANCE_CLASSES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
CLASS_ID_TO_NAME = dict(enumerate(COCO_INSTANCE_CLASSES))

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
        while True:
            i = 10
    image_size = (224, 224)
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    return transform(data)

def filter_empty_lines(text: str) -> Iterator[str]:
    if False:
        return 10
    if len(text.strip()) > 0:
        yield text

class KeyExamplesForEachModelType(beam.DoFn):
    """Duplicate data to run against each model type"""

    def process(self, element: Tuple[torch.Tensor, str]) -> Iterable[Tuple[str, torch.Tensor]]:
        if False:
            return 10
        yield ('v1', element[0])
        yield ('v2', element[0])

class PostProcessor(beam.DoFn):

    def process(self, element: Tuple[str, PredictionResult]) -> Tuple[torch.Tensor, str]:
        if False:
            for i in range(10):
                print('nop')
        (model, prediction_result) = element
        prediction_labels = prediction_result.inference['labels']
        classes = [CLASS_ID_TO_NAME[label.item()] for label in prediction_labels]
        yield (prediction_result.example, f'{model} predictions: {str(classes)}')

class FormatResults(beam.DoFn):

    def process(self, element):
        if False:
            print('Hello World!')
        (_, filename_prediction) = element
        predictions = filename_prediction['predictions']
        v1_predictions = next((p for p in predictions if 'v1 predictions' in p))
        v2_predictions = next((p for p in predictions if 'v2 predictions' in p))
        yield f"{filename_prediction['image_names'][0]} --- {v1_predictions} --- {v2_predictions}"

def parse_known_args(argv):
    if False:
        while True:
            i = 10
    'Parses args for the workflow.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', required=True, help='Path to the text file containing image names.')
    parser.add_argument('--output', dest='output', required=True, help='Path where to save output predictions. text file.')
    parser.add_argument('--model_state_dict_paths', dest='model_state_dict_paths', required=True, help="Comma separated paths to the models' state_dicts. For this example, should include exactly 2 state_dicts corresponding to maskrcnn_resnet50_fpn and maskrcnn_resnet50_fpn_v2 classes.")
    parser.add_argument('--images_dir', help='Path to the directory where images are stored.Not required if image names in the input file have absolute path.')
    return parser.parse_known_args(argv)

def run(argv=None, save_main_session=True, test_pipeline=None) -> PipelineResult:
    if False:
        for i in range(10):
            print('nop')
    '\n  Args:\n    argv: Command line arguments defined for this example.\n    model_params: Parameters passed to the constructor of the model_class.\n                  These will be used to instantiate the model object in the\n                  RunInference API.\n    save_main_session: Used for internal testing.\n    test_pipeline: Used for internal testing.\n  '
    (known_args, pipeline_args) = parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    state_dicts = known_args.model_state_dict_paths.split(',')
    if len(state_dicts) != 2:
        raise AssertionError(f'Expected exactly 2 state_dicts to be supplied, got {len(state_dicts)}')
    mh1 = PytorchModelHandlerTensor(state_dict_path=state_dicts[0], model_class=maskrcnn_resnet50_fpn, model_params={'num_classes': 91})
    mh2 = PytorchModelHandlerTensor(state_dict_path=state_dicts[1], model_class=maskrcnn_resnet50_fpn_v2, model_params={'num_classes': 91})
    model_handler = KeyedModelHandler([KeyModelMapping(['v1'], mh1), KeyModelMapping(['v2'], mh2)])
    pipeline = test_pipeline
    if not test_pipeline:
        pipeline = beam.Pipeline(options=pipeline_options)
    value_filename_pair = pipeline | 'ReadImageNames' >> beam.io.ReadFromText(known_args.input) | 'FilterEmptyLines' >> beam.ParDo(filter_empty_lines) | 'ReadImageData' >> beam.Map(lambda image_name: read_image(image_file_name=image_name, path_to_dir=known_args.images_dir)) | 'PreprocessImages' >> beam.MapTuple(lambda file_name, data: (preprocess_image(data), file_name))
    predictions = value_filename_pair | 'DuplicateData' >> beam.ParDo(KeyExamplesForEachModelType()) | 'PyTorchRunInference' >> RunInference(model_handler) | 'ProcessOutput' >> beam.ParDo(PostProcessor())
    results = {'image_names': value_filename_pair | beam.MapTuple(lambda example, filename: (example.tolist(), filename)), 'predictions': predictions | beam.MapTuple(lambda example, prediction: (example.tolist(), prediction))} | beam.CoGroupByKey()
    _ = results | 'FormatResults' >> beam.ParDo(FormatResults()) | 'WriteOutput' >> beam.io.WriteToText(known_args.output, shard_name_template='', append_trailing_newlines=True)
    result = pipeline.run()
    result.wait_until_finish()
    return result
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()