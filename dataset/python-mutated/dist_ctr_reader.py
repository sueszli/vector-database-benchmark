import logging
import tarfile
import paddle
from paddle.base.log_helper import get_logger
logger = get_logger('paddle', logging.INFO)
DATA_URL = 'http://paddle-ctr-data.bj.bcebos.com/avazu_ctr_data.tgz'
DATA_MD5 = 'c11df99fbd14e53cd4bfa6567344b26e'
'\navazu_ctr_data/train.txt\navazu_ctr_data/infer.txt\navazu_ctr_data/test.txt\navazu_ctr_data/data.meta.txt\n'

def read_data(file_name):
    if False:
        for i in range(10):
            print('nop')
    path = paddle.dataset.common.download(DATA_URL, 'avazu_ctr_data', DATA_MD5)
    tar = tarfile.open(path, 'r:gz')
    tar_info = None
    for member in tar.getmembers():
        if member.name.endswith(file_name):
            tar_info = member
    f = tar.extractfile(tar_info)
    ret_lines = [_.decode('utf-8') for _ in f.readlines()]
    return ret_lines

class TaskMode:
    TRAIN_MODE = 0
    TEST_MODE = 1
    INFER_MODE = 2

    def __init__(self, mode):
        if False:
            while True:
                i = 10
        self.mode = mode

    def is_train(self):
        if False:
            i = 10
            return i + 15
        return self.mode == self.TRAIN_MODE

    def is_test(self):
        if False:
            print('Hello World!')
        return self.mode == self.TEST_MODE

    def is_infer(self):
        if False:
            i = 10
            return i + 15
        return self.mode == self.INFER_MODE

    @staticmethod
    def create_train():
        if False:
            i = 10
            return i + 15
        return TaskMode(TaskMode.TRAIN_MODE)

    @staticmethod
    def create_test():
        if False:
            return 10
        return TaskMode(TaskMode.TEST_MODE)

    @staticmethod
    def create_infer():
        if False:
            while True:
                i = 10
        return TaskMode(TaskMode.INFER_MODE)

class ModelType:
    CLASSIFICATION = 0
    REGRESSION = 1

    def __init__(self, mode):
        if False:
            i = 10
            return i + 15
        self.mode = mode

    def is_classification(self):
        if False:
            return 10
        return self.mode == self.CLASSIFICATION

    def is_regression(self):
        if False:
            for i in range(10):
                print('nop')
        return self.mode == self.REGRESSION

    @staticmethod
    def create_classification():
        if False:
            while True:
                i = 10
        return ModelType(ModelType.CLASSIFICATION)

    @staticmethod
    def create_regression():
        if False:
            while True:
                i = 10
        return ModelType(ModelType.REGRESSION)

def load_dnn_input_record(sent):
    if False:
        print('Hello World!')
    return list(map(int, sent.split()))

def load_lr_input_record(sent):
    if False:
        while True:
            i = 10
    res = []
    for _ in [x.split(':') for x in sent.split()]:
        res.append(int(_[0]))
    return res
feeding_index = {'dnn_input': 0, 'lr_input': 1, 'click': 2}

class Dataset:

    def train(self):
        if False:
            return 10
        '\n        Load trainset.\n        '
        file_name = 'train.txt'
        logger.info('load trainset from %s' % file_name)
        mode = TaskMode.create_train()
        return self._parse_creator(file_name, mode)

    def test(self):
        if False:
            while True:
                i = 10
        '\n        Load testset.\n        '
        file_name = 'test.txt'
        logger.info('load testset from %s' % file_name)
        mode = TaskMode.create_test()
        return self._parse_creator(file_name, mode)

    def infer(self):
        if False:
            while True:
                i = 10
        '\n        Load infer set.\n        '
        file_name = 'infer.txt'
        logger.info('load inferset from %s' % file_name)
        mode = TaskMode.create_infer()
        return self._parse_creator(file_name, mode)

    def _parse_creator(self, file_name, mode):
        if False:
            while True:
                i = 10
        '\n        Parse dataset.\n        '

        def _parse():
            if False:
                while True:
                    i = 10
            data = read_data(file_name)
            for (line_id, line) in enumerate(data):
                fs = line.strip().split('\t')
                dnn_input = load_dnn_input_record(fs[0])
                lr_input = load_lr_input_record(fs[1])
                if not mode.is_infer():
                    click = int(fs[2])
                    yield [dnn_input, lr_input, click]
                else:
                    yield [dnn_input, lr_input]
        return _parse

def load_data_meta():
    if False:
        while True:
            i = 10
    '\n    load data meta info from path, return (dnn_input_dim, lr_input_dim)\n    '
    lines = read_data('data.meta.txt')
    err_info = 'wrong meta format'
    assert len(lines) == 2, err_info
    assert 'dnn_input_dim:' in lines[0] and 'lr_input_dim:' in lines[1], err_info
    res = map(int, [_.split(':')[1] for _ in lines])
    res = list(res)
    logger.info('dnn input dim: %d' % res[0])
    logger.info('lr input dim: %d' % res[1])
    return res