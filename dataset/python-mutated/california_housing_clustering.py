"""A pipeline that uses OnlineClustering transform to group houses
with a similar value together.

This example uses the California Housing Prices dataset from kaggle.
https://www.kaggle.com/datasets/camnugent/california-housing-prices

In the first step of the pipeline, the clustering model is trained
using the OnlineKMeans transform, then the AssignClusterLabels
transform assigns a cluster to each record in the dataset. This
transform makes use of the RunInference API under the hood.

In order to run this example:
1. Download the data from kaggle as csv
2. Run `python california_housing_clustering.py --input <path/to/housing.csv> --checkpoints_path <path/to/checkpoints>`  # pylint: disable=line-too-long
"""
import argparse
import numpy as np
import apache_beam as beam
from apache_beam import pvalue
from apache_beam.dataframe.convert import to_pcollection
from apache_beam.dataframe.io import read_csv
from apache_beam.examples.online_clustering import AssignClusterLabelsInMemoryModel
from apache_beam.examples.online_clustering import OnlineClustering
from apache_beam.examples.online_clustering import OnlineKMeans
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.runners.runner import PipelineResult

def parse_known_args(argv):
    if False:
        i = 10
        return i + 15
    'Parses args for the workflow.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', required=True, help='A csv file containing the data that needs to be clustered.')
    parser.add_argument('--checkpoints_path', dest='checkpoints_path', required=True, help='A path to a directory where model checkpoints can be stored.')
    return parser.parse_known_args(argv)

def run(argv=None, save_main_session=True, test_pipeline=None) -> PipelineResult:
    if False:
        for i in range(10):
            print('nop')
    '\n    Args:\n      argv: Command line arguments defined for this example.\n      save_main_session: Used for internal testing.\n      test_pipeline: Used for internal testing.\n    '
    (known_args, pipeline_args) = parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    pipeline = test_pipeline
    if not test_pipeline:
        pipeline = beam.Pipeline(options=pipeline_options)
    data = pipeline | read_csv(known_args.input)
    features = ['longitude', 'latitude', 'median_income']
    housing_features = to_pcollection(data[features])
    model = housing_features | beam.Map(lambda record: list(record)) | 'Train clustering model' >> OnlineClustering(OnlineKMeans, n_clusters=6, batch_size=256, cluster_args={}, checkpoints_path=known_args.checkpoints_path)
    _ = housing_features | beam.Map(lambda sample: np.array(sample)) | 'RunInference' >> AssignClusterLabelsInMemoryModel(model=pvalue.AsSingleton(model), model_id='kmeans', n_clusters=6, batch_size=512) | beam.Map(print)
    result = pipeline.run()
    result.wait_until_finish()
    return result
if __name__ == '__main__':
    run()