#!/usr/bin/env python
# -*- coding:utf-8 -*-

from py2neo import *
import json
import re
from re import search

def write_json(file_path,data):
    with open(file_path, 'a') as f:    #在代码当前目录生成一个data.json的文件
        json.dump(data, f)

def read_json(file_path):
    with open(file_path, 'r') as f:  # 读取当前目录的json文件并解码成python数据
        data = json.load(f)
        print(data)
        return data

file_path = '/Users/zhangyujuan/graduation/finally.json'
file_data = read_json(file_path)
# file_A_path = 'A.json'
# file_B_path = 'B.json'
# A_all = {}   # 有成份的在Ajson中
# B_all = {}  # 没有成份的放在Bjson中
# for i in file_data:
#     tep = i
#     if not file_data[i]["details"]:
#         # print(file_data[i]["details"])
#         B_all[tep] = {}
#         B_all[tep]["ingredients"] = file_data[i]["ingredients"]
#         B_all[tep]["details"] = file_data[i]["details"]
#         B_all[tep]["price"] = file_data[i]["price"]
#         B_all[tep]["class"] = file_data[i]["class"]
#     else:
#         A_all[tep] = {}
#         A_all[tep]["ingredients"] = file_data[i]["ingredients"]
#         A_all[tep]["details"] = file_data[i]["details"]
#         A_all[tep]["price"] = file_data[i]["price"]
#         A_all[tep]["class"] = file_data[i]["class"]
#
# # write_json(file_B_path, B_all)
# # write_json(file_A_path, A_all)
#
#
# # 分析生成的A和B个有多少个
# B_json = read_json(file_B_path)
# A_json = read_json(file_A_path)
#
# print(len(B_json))  # 202
# print(len(A_json))  # 1160

## ======================================= Engilsh 2 Chinese =================================================
## output :
# 2，data_label : 成份、中文标签
# 3，data_onehot : 成份、onehot

ingre_path = '/Users/zhangyujuan/graduation/ingredients.json'
ingre_data = read_json(ingre_path)

A_path = './data/A.json'  # C
B_path = './data/B.json'  # D

dict_skin ={"Normal":"正常","Dry":"干性","Combination":"混合","Oily":"油性","Sensitive":"敏感"}
list_skin = ["正常","干性","混合","油性","敏感"]

import csv
def create_csv(path,data1,data2):
    tmp = []
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        tmp.append(data1)
        for i in data2:
            tmp.append(i)
        csv_write.writerow(tmp)

def create_csv_data(path,data1):
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        title = ["content","label"]
        csv_write.writerow(title)
        for i in data1:
            csv_write.writerow(i)

def create_csv_data_test(path,data1):
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        title = ["content","正常","干性","混合","油性","敏感"]
        csv_write.writerow(title)
        for i in data1:
            csv_write.writerow(i)

# 先制作 train 和 test的数据集,分成onehot和label
def trans2Chi(file):
    data = read_json(file)
    for i in data:
        tmp = ''
        skin = [0,0,0,0,0]
        chinese_skin = []
        for j in data[i]["ingredients"]:
            if j in ingre_data:
                tmp = tmp+(ingre_data[j]["chinese"])+' '
                # data[i]["ingredients"] = tmp

        # tmp2 转换成one hot 编码格式
        # chinese_skin 中文肤质标签
        for k in data[i]["details"]:
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

# trans2Chi(A_path)


## ======================================= shuffle ==========================================
## output :
# 1，data_onehot_shuffle.csv：成份、onehot、打乱顺序
# 2，data_label_shuffle.csv：成份、标签、打乱顺序

import random
import numpy as np
import pandas as pd

def shuffle_csv(one_hot_file,label_file):
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
# shuffle_csv(one_hot_file,label_file)


# 选择0.8作为训练集，0.2作为测试集
def depart_dataset(csv_file_path):
    csv_file = open(csv_file_path)
    length_file = csv.reader(csv_file)
    length_csv = 1160
    train_length = int(length_csv*0.8)
    test_length = length_csv - train_length
    # print("train_length", train_length)
    train_data = []
    test_data = []
# 前是训练
    # count = 0
    # for one_line in length_file:
    #     # print(one_line)
    #     count += 1
    #     if count < train_length+1:
    #         train_data.append(one_line)
    #     else:
    #         test_data.append(one_line)

    count = 0
    for one_line in length_file:
        # print(one_line)
        count += 1
        if 5*test_length < count <= 6*test_length:
            test_data.append(one_line)
        else:
            train_data.append(one_line)

    print(len(train_data))
    print(len(test_data))
    print(test_data)

    k ='8'

    if "onehot" in csv_file_path:
        create_csv_data_test('./classifier_multi_label_textcnn/data/'+k+'/train_onehot.csv', train_data)
        create_csv_data_test('./classifier_multi_label_textcnn/data/'+k+'/test_onehot.csv', test_data)
    else:
        create_csv_data('./classifier_multi_label_textcnn/data/'+k+'/train.csv', train_data)
        create_csv_data('./classifier_multi_label_textcnn/data/'+k+'/test.csv', test_data)

data_onehot_shuffle_file = './data/data_onehot_shuffle.csv'
data_label_shuffle_file = './data/data_label_shuffle.csv'

# depart_dataset(data_onehot_shuffle_file)
# depart_dataset(data_label_shuffle_file)


# 将待标记数据集转换成csv 格式。
def createB(path,name,data1):
    tmp = [name]
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        tmp.append(data1)
        csv_write.writerow(tmp)

def trans2csv(file):
    data = read_json(file)
    for i in data:
        tmp = ''
        for j in data[i]["ingredients"]:
            if j in ingre_data:
                tmp = tmp+(ingre_data[j]["chinese"])+' '

        createB('./data/B_data.csv', i,tmp)

# trans2csv(B_path)


# run predict.py
# 将B_result数据补充到B.json
def read_csv(file_path):
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
    print(data[0],data[1])
    if data[0] in B_json_data:
        new_de = data[1:]
        B_json_data[data[0]]["details"] = new_de

print(B_json_data)
write_json("B_finally.json",B_json_data)












