import argparse
import os
import sys
import tempfile
from unittest.mock import Mock
import aim
from ludwig.contribs.aim import AimCallback
from tests.integration_tests.utils import category_feature, generate_data, image_feature, run_experiment
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, '..', '..', '..')
sys.path.insert(0, os.path.abspath(PATH_ROOT))

def run(csv_filename):
    if False:
        i = 10
        return i + 15
    callback = AimCallback()
    callback.on_train_init = Mock(side_effect=callback.on_train_init)
    callback.on_train_start = Mock(side_effect=callback.on_train_start)
    with tempfile.TemporaryDirectory() as tmpdir:
        image_dest_folder = os.path.join(tmpdir, 'generated_images')
        input_features = [image_feature(folder=image_dest_folder)]
        output_features = [category_feature(output_feature=True)]
        rel_path = generate_data(input_features, output_features, csv_filename)
        run_experiment(input_features, output_features, dataset=rel_path, callbacks=[callback])
    callback.on_train_init.assert_called()
    callback.on_train_start.assert_called()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-filename', required=True)
    args = parser.parse_args()
    run(args.csv_filename)