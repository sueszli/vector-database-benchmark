from haystack.nodes import FARMReader
import json
import requests
from pathlib import Path
from typing import Union, List
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
download_links = {'squad2': {'train': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json', 'test': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'}, 'squad': {'train': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json', 'test': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'}}

def load_config(path: str) -> dict:
    if False:
        for i in range(10):
            print('nop')
    with open(path, 'r') as f:
        return json.load(f)

def combine_config(configs: dict) -> List[dict]:
    if False:
        return 10
    combinations_list = [[]]
    for (config_key, config) in configs.items():
        if not isinstance(config, list):
            config = [config]
        current_combinations = combinations_list
        combinations_list = []
        for item in config:
            combinations_list += [c + [(config_key, item)] for c in current_combinations]
    combinations = []
    for combination in combinations_list:
        combinations.append(dict(combination))
    return combinations

def download_file(url: str, path: Path):
    if False:
        while True:
            i = 10
    request = requests.get(url, allow_redirects=True)
    with path.open('wb') as f:
        f.write(request.content)

def download_dataset(dataset: Union[dict, str], download_folder: Path):
    if False:
        while True:
            i = 10
    train_file = 'train.json'
    test_file = 'test.json'
    if download_folder.exists():
        assert download_folder.is_dir()
        if (download_folder / train_file).is_file() and (download_folder / test_file).is_file():
            return (train_file, test_file)
    if type(dataset) is str:
        dataset = download_links[dataset]
    train = dataset['train']
    test = dataset['test']
    download_folder.mkdir(parents=True, exist_ok=True)
    download_file(train, download_folder / train_file)
    download_file(test, download_folder / test_file)
    return (train_file, test_file)

def eval(model: FARMReader, download_folder: Path, test_file: str):
    if False:
        while True:
            i = 10
    return model.eval_on_file(data_dir=download_folder, test_filename=test_file)

def train_student(student: dict, download_folder: Path, train_file: str, test_file: str, **kwargs) -> dict:
    if False:
        while True:
            i = 10
    model = FARMReader(model_name_or_path=student['model_name_or_path'])
    model.train(data_dir=download_folder, train_filename=train_file, batch_size=student['batch_size'], caching=True, **kwargs)
    return eval(model, download_folder, test_file)

def train_student_with_distillation(student: dict, teacher: dict, download_folder: Path, train_file: str, test_file: str, **kwargs) -> dict:
    if False:
        print('Hello World!')
    student_model = FARMReader(model_name_or_path=student['model_name_or_path'])
    teacher_model = FARMReader(model_name_or_path=teacher['model_name_or_path'])
    student_model.distil_from(teacher_model, data_dir=download_folder, train_filename=train_file, student_batch_size=student['batch_size'], teacher_batch_size=teacher['batch_size'], caching=True, **kwargs)
    return eval(student_model, download_folder, test_file)

def main():
    if False:
        while True:
            i = 10
    parent = Path(__file__).parent.resolve()
    config = load_config(parent / 'distillation_config.json')
    download_folder = parent / config['download_folder']
    student = config['student_model']
    teacher = config['teacher_model']
    distillation_settings = config['distillation_settings']
    training_settings = config['training_settings']
    logger.info('Downloading dataset')
    (train_file, test_file) = download_dataset(config['dataset'], download_folder)
    results = []
    descriptions = []
    for current_config in combine_config(distillation_settings):
        descriptions.append(f'Results of student with distillation (config: {current_config}')
        logger.info('Training student with distillation (config: %s)', current_config)
        results.append(train_student_with_distillation(student, teacher, download_folder, train_file, test_file, **current_config, **training_settings))
    if config['evaluate_student_without_distillation']:
        logger.info('Training student without distillation as a baseline')
        descriptions.append('Results of student without distillation')
        results.append(train_student(student, download_folder, train_file, test_file, **training_settings))
    if config['evaluate_teacher']:
        logger.info('Evaluating teacher')
        descriptions.append('Results of teacher')
        results.append(eval(FARMReader(model_name_or_path=teacher['model_name_or_path']), download_folder, test_file))
    logger.info('Evaluation results:')
    for (result, description) in zip(results, descriptions):
        logger.info(description)
        logger.info('EM: %s', result['EM'])
        logger.info('F1: %s', result['f1'])
        logger.info('Top n accuracy: %s', result['top_n_accuracy'])
if __name__ == '__main__':
    main()