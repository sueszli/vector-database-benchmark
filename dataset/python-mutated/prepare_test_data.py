import os
import tarfile
import zipfile
from shutil import copyfile
envvar = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'

def _data_copier(src_files, dst_files):
    if False:
        print('Hello World!')
    src_files = [os.path.normpath(os.path.join(os.environ[envvar], *src_file.split('/'))) for src_file in src_files]
    dst_files = [os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), *dst_file.split('/'))) for dst_file in dst_files]
    if not len(src_files) == len(dst_files):
        raise Exception('The length of src and dst should be same')
    for src_dst_file in zip(src_files, dst_files):
        if not os.path.isfile(src_dst_file[1]):
            print('Copying file from: ', src_dst_file[0])
            print('Copying file to: ', src_dst_file[1])
            copyfile(src_dst_file[0], src_dst_file[1])
        else:
            print('Reusing cached file', src_dst_file[1])

def prepare_CIFAR10_data():
    if False:
        while True:
            i = 10
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *'../../../../Examples/Image/DataSets/CIFAR-10'.split('/'))
    base_path = os.path.normpath(base_path)
    if not (os.path.isfile(os.path.join(base_path, 'train_map.txt')) and os.path.isfile(os.path.join(base_path, 'test_map.txt'))):
        base_path_bak = os.path.join(os.environ[envvar], *'Image/CIFAR/v0/cifar-10-batches-py'.split('/'))
        base_path_bak = os.path.normpath(base_path_bak)
        copyfile(os.path.join(base_path_bak, 'train_map.txt'), os.path.join(base_path, 'train_map.txt'))
        copyfile(os.path.join(base_path_bak, 'test_map.txt'), os.path.join(base_path, 'test_map.txt'))
        if not os.path.isdir(os.path.join(base_path, 'cifar-10-batches-py')):
            os.mkdir(os.path.join(base_path, 'cifar-10-batches-py'))
        copyfile(os.path.join(base_path_bak, 'data.zip'), os.path.join(base_path, 'cifar-10-batches-py', 'data.zip'))
        copyfile(os.path.join(base_path_bak, 'CIFAR-10_mean.xml'), os.path.join(base_path, 'CIFAR-10_mean.xml'))
    return base_path

def prepare_ImageNet_data():
    if False:
        print('Hello World!')
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *'../../../../Examples/Image/DataSets/ImageNet'.split('/'))
    base_path = os.path.normpath(base_path)
    if not os.path.isdir(base_path):
        os.mkdir(base_path)
    if not (os.path.isfile(os.path.join(base_path, 'train_map.txt')) and os.path.isfile(os.path.join(base_path, 'val_map.txt'))):
        base_path_bak = os.path.join(os.environ[envvar], *'Image/ImageNet/2012/v0'.split('/'))
        base_path_bak = os.path.normpath(base_path_bak)
        copyfile(os.path.join(base_path_bak, 'val1024_map.txt'), os.path.join(base_path, 'train_map.txt'))
        copyfile(os.path.join(base_path_bak, 'val1024_map.txt'), os.path.join(base_path, 'val_map.txt'))
        copyfile(os.path.join(base_path_bak, 'val1024.zip'), os.path.join(base_path, 'val1024.zip'))
    return base_path

def prepare_MNIST_data():
    if False:
        for i in range(10):
            print('nop')
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *'../../../../Examples/Image/DataSets/MNIST'.split('/'))
    base_path = os.path.normpath(base_path)
    if not os.path.isdir(base_path):
        os.mkdir(base_path)
    if not os.path.isfile(os.path.join(base_path, 'Train-28x28_cntk_text.txt')):
        base_path_bak = os.path.join(os.environ[envvar], *'Image/MNIST'.split('/'))
        base_path_bak = os.path.normpath(base_path_bak)
        copyfile(os.path.join(base_path_bak, 'Train-28x28_cntk_text.txt'), os.path.join(base_path, 'Train-28x28_cntk_text.txt'))
    return base_path

def prepare_Grocery_data():
    if False:
        print('Hello World!')
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *'../../../../Examples/Image/DataSets/Grocery'.split('/'))
    base_path = os.path.normpath(base_path)
    if not os.path.isfile(os.path.join(base_path, 'test.txt')):
        base_path_bak = os.path.join(os.environ[envvar], *'Image/Grocery'.split('/'))
        base_path_bak = os.path.normpath(base_path_bak)
        zip_path = os.path.join(base_path, '..', 'Grocery.zip')
        copyfile(os.path.join(base_path_bak, 'Grocery.zip'), zip_path)
        with zipfile.ZipFile(zip_path) as myzip:
            myzip.extractall(os.path.join(base_path, '..'))
    return base_path

def prepare_fastrcnn_grocery_100_model():
    if False:
        return 10
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *'../../../../PretrainedModels'.split('/'))
    base_path = os.path.normpath(base_path)
    if not os.path.isfile(os.path.join(base_path, 'Fast-RCNN_grocery100.model')):
        base_path_bak = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'], *'PreTrainedModels/FRCN_Grocery/v0'.split('/'))
        base_path_bak = os.path.normpath(base_path_bak)
        model_file_path = os.path.join(base_path, 'Fast-RCNN_grocery100.model')
        copyfile(os.path.join(base_path_bak, 'Fast-RCNN_grocery100.model'), model_file_path)
        print('copied model')
    return base_path

def an4_dataset_directory():
    if False:
        i = 10
        return i + 15
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *'../../../../Examples/Speech/AN4/Data'.split('/'))
    base_path = os.path.normpath(base_path)
    return base_path

def cmudict_dataset_directory():
    if False:
        i = 10
        return i + 15
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *'../../../../Examples/SequenceToSequence/CMUDict/Data'.split('/'))
    base_path = os.path.normpath(base_path)
    return base_path

def prepare_resnet_v1_model():
    if False:
        i = 10
        return i + 15
    src_file = 'PreTrainedModels/ResNet/v1/ResNet_18.model'
    dst_file = 'PretrainedModels/ResNet_18.model'
    _data_copier([src_file], [dst_file])

def prepare_flower_data():
    if False:
        print('Hello World!')
    src_files = ['Image/Flowers/102flowers.tgz', 'Image/Flowers/imagelabels.mat', 'Image/Flowers/imagelabels.mat']
    dst_files = ['Examples/Image/DataSets/Flowers/102flowers.tgz', 'Examples/Image/DataSets/Flowers/imagelabels.mat', 'Examples/Image/DataSets/Flowers/imagelabels.mat']
    _data_copier(src_files, dst_files)

def prepare_animals_data():
    if False:
        print('Hello World!')
    src_file = 'Image/Animals/Animals.zip'
    dst_file = 'Examples/Image/DataSets/Animals/Animals.zip'
    _data_copier([src_file], [dst_file])

def prepare_alexnet_v0_model():
    if False:
        for i in range(10):
            print('nop')
    local_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *'../../../../PretrainedModels'.split('/'))
    local_base_path = os.path.normpath(local_base_path)
    model_file = os.path.join(local_base_path, 'AlexNet.model')
    if not os.path.isfile(model_file):
        external_model_path = os.path.join(os.environ[envvar], 'PreTrainedModels', 'AlexNet', 'v0', 'AlexNet.model')
        copyfile(external_model_path, model_file)
    model_file = os.path.join(local_base_path, 'AlexNet_ImageNet_Caffe.model')
    if not os.path.isfile(model_file):
        external_model_path = os.path.join(os.environ[envvar], 'PreTrainedModels', 'AlexNet', 'v1', 'AlexNet_ImageNet_Caffe.model')
        copyfile(external_model_path, model_file)
    return local_base_path

def prepare_UCF11_data():
    if False:
        for i in range(10):
            print('nop')
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *'../../../../Examples/Video/DataSets/UCF11'.split('/'))
    base_path = os.path.normpath(base_path)
    if not os.path.isfile(os.path.join(base_path, 'test_map.csv')):
        tar_path = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'], *'DataSets/UCF11-v0.tar'.split('/'))
        with tarfile.TarFile(tar_path) as mytar:
            mytar.extractall(base_path)

def prepare_WordLMWithSampledSoftmax_ptb_data():
    if False:
        for i in range(10):
            print('nop')
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'Examples', 'Text', 'WordLMWithSampledSoftmax', 'ptb')
    base_path = os.path.normpath(base_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    external_data_path = os.path.join(os.environ[envvar], 'Text', 'WordLMWithSampledSoftmax_ptb')
    src_files = ['test.txt', 'token2freq.txt', 'token2id.txt', 'train.txt', 'valid.txt', 'vocab.txt', 'freq.txt']
    for src_file in src_files:
        if os.path.isfile(os.path.join(base_path, src_file)):
            continue
        copyfile(os.path.join(external_data_path, src_file), os.path.join(base_path, src_file))