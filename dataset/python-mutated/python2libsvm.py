from __future__ import print_function
import os
import sklearn.datasets as datasets

def get_data(file_input, separator='\t'):
    if False:
        while True:
            i = 10
    if 'libsvm' not in file_input:
        file_input = other2libsvm(file_input, separator)
    data = datasets.load_svmlight_file(file_input)
    return (data[0], data[1])

def other2libsvm(file_name, separator='\t'):
    if False:
        for i in range(10):
            print('nop')
    libsvm_name = file_name.replace('.txt', '.libsvm_tmp')
    libsvm_data = open(libsvm_name, 'w')
    file_data = open(file_name, 'r')
    for line in file_data.readlines():
        features = line.strip().split(separator)
        class_data = features[-1]
        svm_format = ''
        for i in range(len(features) - 1):
            svm_format += ' %d:%s' % (i + 1, features[i])
        svm_format = '%s%s\n' % (class_data, svm_format)
        libsvm_data.write(svm_format)
    file_data.close()
    libsvm_data.close()
    return libsvm_name

def dump_data(x, y, file_output):
    if False:
        i = 10
        return i + 15
    datasets.dump_svmlight_file(x, y, file_output)
    os.remove('%s_tmp' % file_output)
if __name__ == '__main__':
    file_input = 'data/7.AdaBoost/horseColicTest2.txt'
    file_output = 'data/7.AdaBoost/horseColicTest2.libsvm'
    (x, y) = get_data(file_input, separator='\t')
    print(x[3, :])
    print(y)
    dump_data(x, y, file_output)