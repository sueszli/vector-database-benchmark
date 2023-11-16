import argparse
import os
import sys
import tempfile
from unittest.mock import Mock, patch
import comet_ml
from ludwig.api import LudwigModel
from ludwig.constants import BATCH_SIZE, TRAINER
from ludwig.contribs.comet import CometCallback
os.environ['COMET_API_KEY'] = 'key'
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, '..', '..', '..')
sys.path.insert(0, os.path.abspath(PATH_ROOT))
from tests.integration_tests.utils import category_feature, generate_data, image_feature
parser = argparse.ArgumentParser()
parser.add_argument('--csv-filename', required=True)

def run(csv_filename):
    if False:
        i = 10
        return i + 15
    with tempfile.TemporaryDirectory() as tmpdir:
        image_dest_folder = os.path.join(tmpdir, 'generated_images')
        input_features = [image_feature(folder=image_dest_folder)]
        output_features = [category_feature(output_feature=True)]
        data_csv = generate_data(input_features, output_features, csv_filename)
        config = {'input_features': input_features, 'output_features': output_features, 'combiner': {'type': 'concat', 'output_size': 14}, TRAINER: {'epochs': 2, BATCH_SIZE: 128}}
        callback = CometCallback()
        model = LudwigModel(config, callbacks=[callback])
        callback.on_train_init = Mock(side_effect=callback.on_train_init)
        callback.on_train_start = Mock(side_effect=callback.on_train_start)
        with patch('comet_ml.Experiment.log_asset_data') as mock_log_asset_data:
            (_, _, _) = model.train(dataset=data_csv, output_directory=os.path.join(tmpdir, 'output'))
            model.predict(dataset=data_csv)
    assert callback.cometml_experiment is not None
    callback.on_train_init.assert_called()
    callback.on_train_start.assert_called()
    mock_log_asset_data.assert_called()
if __name__ == '__main__':
    args = parser.parse_args()
    run(args.csv_filename)