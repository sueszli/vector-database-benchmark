"""This does model predictions for different locations at a range of years."""
from __future__ import annotations
from collections.abc import Iterable, Sequence
import csv
from typing import NamedTuple
import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import RunInference
from apache_beam.options.pipeline_options import PipelineOptions
import numpy as np
from serving import data
PATCH_SIZE = 512
LOCATIONS_FILE = 'predict-locations.csv'
MAX_REQUESTS = 20
YEARS = [2016, 2017, 2018, 2019, 2020, 2021]

class Location(NamedTuple):
    name: str
    year: int
    point: tuple[float, float]

def get_inputs(location: Location, patch_size: int=PATCH_SIZE, predictions_path: str='predictions') -> tuple[str, np.ndarray]:
    if False:
        i = 10
        return i + 15
    'Get an inputs patch to predict.\n\n    Args:\n        location: A name, year, and (longitude, latitude) point.\n        patch_size: Size in pixels of the surrounding square patch.\n        predictions_path: Directory path to save prediction results.\n\n    Returns: A (file_path_name, inputs_patch) pair.\n    '
    data.ee_init()
    path = FileSystems.join(predictions_path, location.name, str(location.year))
    inputs = data.get_input_patch(location.year, location.point, patch_size)
    return (path, inputs)

def write_numpy(path: str, data: np.ndarray, label: str='data') -> str:
    if False:
        print('Hello World!')
    'Writes the prediction results into a compressed NumPy file (*.npz).\n\n    Args:\n        path: File path prefix to save to.\n        data: NumPy array holding the data.\n        label: Used as a suffix to the filename, and a key for the NumPy file.\n\n    Returns: The file name where the data was saved to.\n    '
    filename = f'{path}-{label}.npz'
    with FileSystems.create(filename) as f:
        np.savez_compressed(f, **{label: data})
    logging.info(filename)
    return filename

def run_tensorflow(locations: Iterable[Location], model_path: str, predictions_path: str, patch_size: int=PATCH_SIZE, max_requests: int=MAX_REQUESTS, beam_args: list[str] | None=None) -> None:
    if False:
        print('Hello World!')
    'Runs an Apache Beam pipeline to do batch predictions.\n\n    This fetches data from Earth Engine and does batch prediction on the data.\n    We use `max_requests` to limit the number of concurrent requests to Earth Engine\n    to avoid quota issues. You can request for an increas of quota if you need it.\n\n    Args:\n        locations: A collection of name, point, and year.\n        model_path: Directory path to load the trained model from.\n        predictions_path: Directory path to save prediction results.\n        patch_size: Size in pixels of the surrounding square patch.\n        max_requests: Limit the number of concurrent requests to Earth Engine.\n        beam_args: Apache Beam command line arguments to parse as pipeline options.\n    '
    import tensorflow as tf

    class LandCoverModel(ModelHandler[np.ndarray, np.ndarray, tf.keras.Model]):

        def load_model(self) -> tf.keras.Model:
            if False:
                for i in range(10):
                    print('nop')
            return tf.keras.models.load_model(model_path)

        def run_inference(self, batch: Sequence[np.ndarray], model: tf.keras.Model, inference_args: dict | None=None) -> Iterable[np.ndarray]:
            if False:
                print('Hello World!')
            probabilities = model.predict(np.stack(batch))
            predictions = probabilities.argmax(axis=-1).astype(np.uint8)
            return predictions[:, :, :, None]
    model_handler = KeyedModelHandler(LandCoverModel())
    beam_options = PipelineOptions(beam_args, save_main_session=True, setup_file='./setup.py', max_num_workers=max_requests, direct_num_workers=max(max_requests, 20), disk_size_gb=50)
    with beam.Pipeline(options=beam_options) as pipeline:
        inputs = pipeline | 'Locations' >> beam.Create(locations) | 'Get inputs' >> beam.Map(get_inputs, patch_size, predictions_path)
        predictions = inputs | 'RunInference' >> RunInference(model_handler)
        inputs | 'Write inputs' >> beam.MapTuple(write_numpy, 'inputs')
        predictions | 'Write predictions' >> beam.MapTuple(write_numpy, 'predictions')
if __name__ == '__main__':
    import argparse
    import logging
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('framework', choices=['tensorflow'])
    parser.add_argument('--model-path', required=True, help='Directory path to load the trained model from.')
    parser.add_argument('--predictions-path', required=True, help='Directory path to save prediction results.')
    parser.add_argument('--locations-file', default=LOCATIONS_FILE, help='CSV file with the location names and points to predict.')
    parser.add_argument('--patch-size', default=PATCH_SIZE, type=int, help='Size in pixels of the surrounding square patch.')
    parser.add_argument('--max-requests', default=MAX_REQUESTS, type=int, help='Limit the number of concurrent requests to Earth Engine.')
    (args, beam_args) = parser.parse_known_args()
    with open(args.locations_file) as f:
        locations = [Location(row['name'], year, (float(row['lon']), float(row['lat']))) for row in csv.DictReader(f) for year in YEARS]
    if args.framework == 'tensorflow':
        run_tensorflow(locations=locations, model_path=args.model_path, predictions_path=args.predictions_path, patch_size=args.patch_size, max_requests=args.max_requests, beam_args=beam_args)
    else:
        raise ValueError(f'framework not supported: {args.framework}')