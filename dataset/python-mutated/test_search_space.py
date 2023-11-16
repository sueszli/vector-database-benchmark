import json
from pathlib import Path
import yaml
from nni.experiment.config import ExperimentConfig, AlgorithmConfig, LocalConfig
config = ExperimentConfig(search_space_file='', trial_command='echo hello', trial_concurrency=1, tuner=AlgorithmConfig(name='randomm'), training_service=LocalConfig())
space_correct = {'pool_type': {'_type': 'choice', '_value': ['max', 'min', 'avg']}, '学习率': {'_type': 'loguniform', '_value': [1e-07, 0.1]}}
formats = [('ss_tab.json', 'JSON (tabs + scientific notation)'), ('ss_comma.json', 'JSON with extra comma'), ('ss.yaml', 'YAML')]

def test_search_space():
    if False:
        return 10
    for (space_file, description) in formats:
        try:
            config.search_space_file = Path(__file__).parent / 'assets' / space_file
            space = config.json()['searchSpace']
            assert space == space_correct
        except Exception as e:
            print('Failed to load search space format: ' + description)
            raise e
if __name__ == '__main__':
    test_search_space()