from py2neo import *
import json
import re
from re import search

def write_json(file_path, data):
    if False:
        print('Hello World!')
    with open(file_path, 'a') as f:
        json.dump(data, f)

def read_json(file_path):
    if False:
        while True:
            i = 10
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(data)
        return data
file_path = '/Users/zhangyujuan/graduation/finally.json'
file_data = read_json(file_path)
ingre_path = '/Users/zhangyujuan/graduation/ingredients.json'
ingre_data = read_json(ingre_path)
A_path = './data/A.json'
B_path = './data/B.json'
dict_skin = {'Normal': '正常', 'Dry': '干性', 'Combination': '混合', 'Oily': '油性', 'Sensitive': '敏感'}
list_skin = ['正常', '干性', '混合', '油性', '敏感']
import csv

def create_csv(path, data1, data2):
    if False:
        i = 10
        return i + 15
    tmp = []
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        tmp.append(data1)
        for i in data2:
            tmp.append(i)
        csv_write.writerow(tmp)

def create_csv_data(path, data1):
    if False:
        return 10
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        title = ['content', 'label']
        csv_write.writerow(title)
        for i in data1:
            csv_write.writerow(i)

def create_csv_data_test(path, data1):
    if False:
        print('Hello World!')
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        title = ['content', '正常', '干性', '混合', '油性', '敏感']
        csv_write.writerow(title)
        for i in data1:
            csv_write.writerow(i)

def trans2Chi(file):
    if False:
        return 10
    data = read_json(file)
    for i in data:
        tmp = ''
        skin = [0, 0, 0, 0, 0]
        chinese_skin = []
        for j in data[i]['ingredients']:
            if j in ingre_data:
                tmp = tmp + ingre_data[j]['chinese'] + ' '
        for k in data[i]['details']:
            if k in dict_skin:
                a = list_skin.index(dict_skin[k])
                chinese_skin.append(dict_skin[k])
                skin[a] = 1
        tmp2 = ''
        for m in skin:
            tmp2 = tmp2 + str(m)
        print(i)
        print(chinese_skin)
        create_csv('./data/data_label.csv', tmp, chinese_skin)
        create_csv('./data/data_onehot.csv', tmp, tmp2)
import random
import numpy as np
import pandas as pd

def shuffle_csv(one_hot_file, label_file):
    if False:
        for i in range(10):
            print('nop')
    with open(one_hot_file) as one_hot_file:
        one_hot_file = csv.reader(one_hot_file)
        one_hot_data = []
        for one_line in one_hot_file:
            one_hot_data.append(one_line)
    with open(label_file) as label_file:
        label_file = csv.reader(label_file)
        label_data = []
        for one_line in label_file:
            label_data.append(one_line)
    state = np.random.get_state()
    np.random.shuffle(one_hot_data)
    np.random.set_state(state)
    np.random.shuffle(label_data)
    create_csv_data('./data/data_onehot_shuffle.csv', one_hot_data)
    create_csv_data('./data/data_label_shuffle.csv', label_data)
one_hot_file = '/Users/zhangyujuan/graduation/data/data_onehot.csv'
label_file = '/Users/zhangyujuan/graduation/data/data_label.csv'

def depart_dataset(csv_file_path):
    if False:
        print('Hello World!')
    csv_file = open(csv_file_path)
    length_file = csv.reader(csv_file)
    length_csv = 1160
    train_length = int(length_csv * 0.8)
    test_length = length_csv - train_length
    train_data = []
    test_data = []
    count = 0
    for one_line in length_file:
        count += 1
        if 5 * test_length < count <= 6 * test_length:
            test_data.append(one_line)
        else:
            train_data.append(one_line)
    print(len(train_data))
    print(len(test_data))
    print(test_data)
    k = '8'
    if 'onehot' in csv_file_path:
        create_csv_data_test('./classifier_multi_label_textcnn/data/' + k + '/train_onehot.csv', train_data)
        create_csv_data_test('./classifier_multi_label_textcnn/data/' + k + '/test_onehot.csv', test_data)
    else:
        create_csv_data('./classifier_multi_label_textcnn/data/' + k + '/train.csv', train_data)
        create_csv_data('./classifier_multi_label_textcnn/data/' + k + '/test.csv', test_data)
data_onehot_shuffle_file = './data/data_onehot_shuffle.csv'
data_label_shuffle_file = './data/data_label_shuffle.csv'

def createB(path, name, data1):
    if False:
        print('Hello World!')
    tmp = [name]
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        tmp.append(data1)
        csv_write.writerow(tmp)

def trans2csv(file):
    if False:
        return 10
    data = read_json(file)
    for i in data:
        tmp = ''
        for j in data[i]['ingredients']:
            if j in ingre_data:
                tmp = tmp + ingre_data[j]['chinese'] + ' '
        createB('./data/B_data.csv', i, tmp)

def read_csv(file_path):
    if False:
        for i in range(10):
            print('nop')
    with open(file_path) as f:
        label_file = csv.reader(f)
        data = []
        for online in label_file:
            data.append(online)
        return data
B_result_file = '/Users/zhangyujuan/graduation/classifier_multi_label_textcnn/B_result.csv'
B_result_data = read_csv(B_result_file)
B_json_file = '/Users/zhangyujuan/graduation/data/B.json'
B_json_data = read_json(B_json_file)
for data in B_result_data:
    print(data[0], data[1])
    if data[0] in B_json_data:
        new_de = data[1:]
        B_json_data[data[0]]['details'] = new_de
print(B_json_data)
write_json('B_finally.json', B_json_data)