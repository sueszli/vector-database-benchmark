"""
This example demonstrates how to use MLTransform.
MLTransform is a PTransform that applies multiple data transformations on the
incoming data.

This example computes the vocabulary on the incoming data. Then, it computes
the TF-IDF of the incoming data using the vocabulary computed in the previous
step.

1. ComputeAndApplyVocabulary computes the vocabulary on the incoming data and
   overrides the incoming data with the vocabulary indices.
2. TFIDF computes the TF-IDF of the incoming data using the vocabulary and
    provides vocab_index and tf-idf weights. vocab_index is suffixed with
    '_vocab_index' and tf-idf weights are suffixed with '_tfidf' to the
    original column name(which is the output of ComputeAndApplyVocabulary).

MLTransform produces artifacts, for example: ComputeAndApplyVocabulary produces
a text file that contains vocabulary which is saved in `artifact_location`.
ComputeAndApplyVocabulary outputs vocab indices associated with the saved vocab
file. This mode of MLTransform is called artifact `produce` mode.
This will be useful when the data is preprocessed before ML model training.

The second mode of MLTransform is artifact `consume` mode. In this mode, the
transformations are applied on the incoming data using the artifacts produced
by the previous run of MLTransform. This mode will be useful when the data is
preprocessed before ML model inference.
"""
import argparse
import logging
import tempfile
import apache_beam as beam
from apache_beam.ml.transforms.base import MLTransform
from apache_beam.ml.transforms.tft import TFIDF
from apache_beam.ml.transforms.tft import ComputeAndApplyVocabulary
from apache_beam.ml.transforms.utils import ArtifactsFetcher

def parse_args():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact_location', type=str, default='')
    return parser.parse_known_args()

def preprocess_data_for_ml_training(train_data, args):
    if False:
        print('Hello World!')
    '\n  Preprocess the data for ML training. This method runs a pipeline to\n  preprocess the data needed for ML training. It produces artifacts that can\n  be used for ML inference later.\n  '
    with beam.Pipeline() as p:
        train_data_pcoll = p | 'CreateData' >> beam.Create(train_data)
        transformed_data_pcoll = train_data_pcoll | 'MLTransform' >> MLTransform(write_artifact_location=args.artifact_location).with_transform(ComputeAndApplyVocabulary(columns=['x'])).with_transform(TFIDF(columns=['x']))
        _ = transformed_data_pcoll | beam.Map(logging.info)

def preprocess_data_for_ml_inference(test_data, args):
    if False:
        while True:
            i = 10
    '\n  Preprocess the data for ML inference. This method runs a pipeline to\n  preprocess the data needed for ML inference. It consumes the artifacts\n  produced during the preprocessing stage for ML training.\n  '
    with beam.Pipeline() as p:
        test_data_pcoll = p | beam.Create(test_data)
        transformed_data_pcoll = test_data_pcoll | 'MLTransformOnTestData' >> MLTransform(read_artifact_location=args.artifact_location)
        _ = transformed_data_pcoll | beam.Map(logging.info)

def run(args):
    if False:
        return 10
    '\n  This example demonstrates how to use MLTransform in ML workflow.\n  1. Preprocess the data for ML training.\n  2. Do some ML model training.\n  3. Preprocess the data for ML inference.\n\n  training and inference on ML modes are not shown in this example.\n  This example only shows how to use MLTransform for preparing data for ML\n  training and inference.\n  '
    train_data = [dict(x=["Let's", 'go', 'to', 'the', 'park']), dict(x=['I', 'enjoy', 'going', 'to', 'the', 'park']), dict(x=['I', 'enjoy', 'reading', 'books']), dict(x=['Beam', 'can', 'be', 'fun']), dict(x=['The', 'weather', 'is', 'really', 'nice', 'today']), dict(x=['I', 'love', 'to', 'go', 'to', 'the', 'park']), dict(x=['I', 'love', 'to', 'read', 'books']), dict(x=['I', 'love', 'to', 'program'])]
    test_data = [dict(x=['I', 'love', 'books']), dict(x=['I', 'love', 'Apache', 'Beam'])]
    preprocess_data_for_ml_training(train_data, args=args)
    preprocess_data_for_ml_inference(test_data, args=args)
    artifacts_fetcher = ArtifactsFetcher(artifact_location=args.artifact_location)
    vocab_list = artifacts_fetcher.get_vocab_list()
    assert vocab_list[22] == 'Beam'
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    (args, pipeline_args) = parse_args()
    if args.artifact_location == '':
        args.artifact_location = tempfile.mkdtemp()
    run(args)