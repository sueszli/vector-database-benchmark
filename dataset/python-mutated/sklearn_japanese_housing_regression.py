"""A pipeline that uses RunInference API on a regression about housing prices.

This example uses the japanese housing data from kaggle.
https://www.kaggle.com/datasets/nishiodens/japan-real-estate-transaction-prices

Since the data has missing fields, this example illustrates how to split
data and assign it to the models that are trained on different subsets of
features. The predictions are then recombined.

In order to set this example up, you will need two things.
1. Build models (or use ours) and reference those via the model directory.
2. Download the data from kaggle and host it.
"""
import argparse
import os
from typing import Iterable
import pandas
import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.sklearn_inference import ModelFileType
from apache_beam.ml.inference.sklearn_inference import SklearnModelHandlerPandas
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.runners.runner import PipelineResult
MODELS = [{'name': 'all_features', 'required_features': ['Area', 'Year', 'MinTimeToNearestStation', 'MaxTimeToNearestStation', 'TotalFloorArea', 'Frontage', 'Breadth', 'BuildingYear']}, {'name': 'floor_area', 'required_features': ['Area', 'Year', 'TotalFloorArea']}, {'name': 'stations', 'required_features': ['Area', 'Year', 'MinTimeToNearestStation', 'MaxTimeToNearestStation']}, {'name': 'no_features', 'required_features': ['Area', 'Year']}]

def sort_by_features(dataframe, max_size):
    if False:
        return 10
    ' Returns an index to a model, based on available data.'
    for (i, model) in enumerate(MODELS):
        required_features = dataframe[model['required_features']]
        if required_features.notnull().all().all():
            return i
    return -1

class LoadDataframe(beam.DoFn):

    def process(self, file_name: str) -> Iterable[pandas.DataFrame]:
        if False:
            while True:
                i = 10
        ' Loads data files as a pandas dataframe.'
        file = FileSystems.open(file_name, 'rb')
        dataframe = pandas.read_csv(file)
        for i in range(dataframe.shape[0]):
            yield dataframe.iloc[[i]]

def report_predictions(prediction_result):
    if False:
        while True:
            i = 10
    true_result = prediction_result.example['TradePrice'].values[0]
    inference = prediction_result.inference
    return 'True Price %.0f, Predicted Price %.0f' % (true_result, inference)

def parse_known_args(argv):
    if False:
        return 10
    'Parses args for the workflow.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', required=True, help='A single or comma-separated list of files or uris.')
    parser.add_argument('--model_path', dest='model_path', required=True, help='A path from where all models can be read.')
    parser.add_argument('--output', dest='output', required=True, help='Path to save output predictions.')
    return parser.parse_known_args(argv)

def inference_transform(model_name, model_path):
    if False:
        print('Hello World!')
    'Returns a RunInference transform.'
    model_filename = model_path + model_name + '.pickle'
    model_loader = SklearnModelHandlerPandas(model_file_type=ModelFileType.PICKLE, model_uri=model_filename)
    transform_name = 'RunInference ' + model_name
    return transform_name >> RunInference(model_loader)

def run(argv=None, save_main_session=True, test_pipeline=None) -> PipelineResult:
    if False:
        print('Hello World!')
    'Entry point. Defines and runs the pipeline.'
    (known_args, pipeline_args) = parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    requirements_dir = os.path.dirname(os.path.realpath(__file__))
    pipeline_options.view_as(SetupOptions).requirements_file = f'{requirements_dir}/sklearn_examples_requirements.txt'
    pipeline = test_pipeline
    if not test_pipeline:
        pipeline = beam.Pipeline(options=pipeline_options)
    file_names = pipeline | 'FileNames' >> beam.Create(known_args.input.split(','))
    loaded_data = file_names | beam.ParDo(LoadDataframe())
    [all, floor_area, stations, no_features] = loaded_data | 'Partition' >> beam.Partition(sort_by_features, len(MODELS))
    model_path = known_args.model_path
    prediction_1 = all | inference_transform('all_features', model_path)
    prediction_2 = floor_area | inference_transform('floor_area', model_path)
    prediction_3 = stations | inference_transform('stations', model_path)
    prediction_4 = no_features | inference_transform('no_features', model_path)
    all_predictions = (prediction_1, prediction_2, prediction_3, prediction_4)
    flattened_predictions = all_predictions | 'Flatten' >> beam.Flatten()
    prediction_report = flattened_predictions | 'AllPredictions' >> beam.Map(report_predictions)
    _ = prediction_report | 'WriteOutput' >> beam.io.WriteToText(known_args.output, append_trailing_newlines=True, shard_name_template='')
    result = pipeline.run()
    result.wait_until_finish()
    return result
if __name__ == '__main__':
    run()