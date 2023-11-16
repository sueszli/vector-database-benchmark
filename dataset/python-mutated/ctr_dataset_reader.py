import logging
import os
import random
import tarfile
import tempfile
import warnings
import paddle
from paddle.distributed import fleet
logging.basicConfig()
logger = logging.getLogger('paddle')
logger.setLevel(logging.INFO)
DATA_URL = 'http://paddle-ctr-data.bj.bcebos.com/avazu_ctr_data.tgz'
DATA_MD5 = 'c11df99fbd14e53cd4bfa6567344b26e'
'\navazu_ctr_data/train.txt\navazu_ctr_data/infer.txt\navazu_ctr_data/test.txt\navazu_ctr_data/data.meta.txt\n'

def download_file():
    if False:
        for i in range(10):
            print('nop')
    file_name = 'avazu_ctr_data'
    path = paddle.dataset.common.download(DATA_URL, file_name, DATA_MD5)
    dir_name = os.path.dirname(path)
    text_file_dir_name = os.path.join(dir_name, file_name)
    if not os.path.exists(text_file_dir_name):
        tar = tarfile.open(path, 'r:gz')
        tar.extractall(dir_name)
    return text_file_dir_name

def load_dnn_input_record(sent):
    if False:
        return 10
    return list(map(int, sent.split()))

def load_lr_input_record(sent):
    if False:
        while True:
            i = 10
    res = []
    for _ in [x.split(':') for x in sent.split()]:
        res.append(int(_[0]) % 10000)
    return res

class CtrReader:

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def _reader_creator(self, filelist):
        if False:
            return 10

        def get_rand(low=0.0, high=1.0):
            if False:
                print('Hello World!')
            return random.random()

        def reader():
            if False:
                print('Hello World!')
            for file in filelist:
                with open(file, 'r') as f:
                    for line in f:
                        if get_rand() < 0.05:
                            fs = line.strip().split('\t')
                            dnn_input = load_dnn_input_record(fs[0])
                            lr_input = load_lr_input_record(fs[1])
                            click = [int(fs[2])]
                            yield ([dnn_input] + [lr_input] + [click])
        return reader

class DatasetCtrReader(fleet.MultiSlotDataGenerator):

    def generate_sample(self, line):
        if False:
            print('Hello World!')

        def get_rand(low=0.0, high=1.0):
            if False:
                while True:
                    i = 10
            return random.random()

        def iter():
            if False:
                while True:
                    i = 10
            if get_rand() < 0.05:
                fs = line.strip().split('\t')
                dnn_input = load_dnn_input_record(fs[0])
                lr_input = load_lr_input_record(fs[1])
                click = [int(fs[2])]
                yield (('dnn_data', dnn_input), ('lr_data', lr_input), ('click', click))
        return iter

def prepare_data():
    if False:
        print('Hello World!')
    '\n    load data meta info from path, return (dnn_input_dim, lr_input_dim)\n    '
    file_dir_name = download_file()
    meta_file_path = os.path.join(file_dir_name, 'data.meta.txt')
    train_file_path = os.path.join(file_dir_name, 'train.txt')
    with open(meta_file_path, 'r') as f:
        lines = f.readlines()
    err_info = 'wrong meta format'
    assert len(lines) == 2, err_info
    assert 'dnn_input_dim:' in lines[0] and 'lr_input_dim:' in lines[1], err_info
    res = map(int, [_.split(':')[1] for _ in lines])
    res = list(res)
    dnn_input_dim = res[0]
    lr_input_dim = res[1]
    logger.info('dnn input dim: %d' % dnn_input_dim)
    logger.info('lr input dim: %d' % lr_input_dim)
    return (dnn_input_dim, lr_input_dim, train_file_path)

def gen_fake_line(dnn_data_num=7, dnn_data_range=100000.0, lr_data_num=5, lr_data_range=100000.0):
    if False:
        return 10
    line = ''
    for index in range(dnn_data_num):
        data = str(random.randint(0, dnn_data_range - 1))
        if index < dnn_data_num - 1:
            data += ' '
        line += data
    line += '\t'
    for index in range(lr_data_num):
        data = str(random.randint(0, lr_data_range - 1)) + ':' + str(1)
        if index < lr_data_num - 1:
            data += ' '
        line += data
    line += '\t'
    line += str(random.randint(0, 1))
    line += '\n'
    return line

def gen_zero_line(dnn_data_num=7, lr_data_num=5):
    if False:
        i = 10
        return i + 15
    line = ''
    for index in range(dnn_data_num):
        data = str(0)
        if index < dnn_data_num - 1:
            data += ' '
        line += data
    line += '\t'
    for index in range(lr_data_num):
        data = str(0) + ':' + str(1)
        if index < lr_data_num - 1:
            data += ' '
        line += data
    line += '\t'
    line += str(random.randint(0, 1))
    line += '\n'
    return line

def prepare_fake_data(file_nums=4, file_lines=500):
    if False:
        return 10
    '\n    Create fake data with same type as avazu_ctr_data\n    '
    file_dir = tempfile.mkdtemp()
    warnings.warn(f'Fake data write in {file_dir}')
    for file_index in range(file_nums):
        with open(os.path.join(file_dir, f'ctr_train_data_part_{file_index}'), 'w+') as fin:
            file_str = ''
            file_str += gen_zero_line()
            for line_index in range(file_lines - 1):
                file_str += gen_fake_line()
            fin.write(file_str)
            warnings.warn(f'Write done ctr_train_data_part_{file_index}')
    file_list = [os.path.join(file_dir, x) for x in os.listdir(file_dir)]
    assert len(file_list) == file_nums
    return file_list
if __name__ == '__main__':
    pairwise_reader = DatasetCtrReader()
    pairwise_reader.run_from_stdin()