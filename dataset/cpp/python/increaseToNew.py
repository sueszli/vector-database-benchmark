#!/usr/bin/env python
# -*- coding:utf-8 -*-

# =================================== 希望将数据集整合成json格式 ======================================================
# import json
# import csv
# import  pandas as pd
# import re
#
# csv_file_path = '/Users/zhangyujuan/graduation/Dataset-sephora.csv'
#
# csvfile = open(csv_file_path, encoding='utf-8')
# csv_file = pd.read_csv(csvfile)  # [1750 rows x 21 columns]
# json_file = "details.txt"
# skin =dict()
#
# def wrtie_json_data(data,file):
#     json.dump(data,file,ensure_ascii=False)
# empty = 0
# total = 0
# for index,row in csv_file.iterrows():
#     total+=1
#     tmp = dict()
#     raw_details = row["details"]
#     # raw_details = list(raw_details)
#     start = r'Which skin type is it good for ?'
#     end = r'What it is:'
#     a = start+'(.*?)'+end
#     ss = re.findall(a, raw_details, re.S)
#     if ss:
#         tt = ss[0]
#         ss = tt.split()
#         ss.remove("?")
#     else:
#         start2 = r"Skin Type: "
#         end2 = r"Skincare Concerns"
#         end3 = r"Formulation"
#         b = start2 + '(.*?)' + end2
#         ss = re.findall(b, raw_details, re.S)
#         if ss:
#             tt = ss[0]
#             ss = tt.replace('-','').replace('and','').split()
#         else:
#             c = start2 + '(.*?)' + end3
#             ss = re.findall(c, raw_details, re.S)
#             if ss:
#                 tt = ss[0]
#                 ss = tt.replace('-', '').replace('and', '').split()
#     if not ss:
#         empty+=1
#     tmp["details"] = ss
#     skin[row["name"]] = tmp
# print("empty ",empty)
# print("total ",total)
# skin = str(skin)
# strdict3 = skin.replace("\'", "\"")
# print(strdict3)


# ============================================ 制作： New_Datasets_2.json 数据集 =======================================================

# =================================== 增加new name 以及 ingredients数据、删除没有成份的数据 ======================================================
import json


# test_file_path = '/Users/zhangyujuan/graduation/test.json'
#
# def write_json(file_path,data):
#     with open(file_path, 'w') as f:    #在代码当前目录生成一个data.json的文件
#         json.dump(data, f)
#
# def read_json(file_path):
#     with open(file_path, 'r') as f:  # 读取当前目录的json文件并解码成python数据
#         data = json.load(f)
#         print(data)
#         return data
#
# def list_all_dict(dict_a):
#     if isinstance(dict_a,dict) : #使用isinstance检测数据类型
#         for x in (dict_a):
#             # temp_value = dict_a[temp_key]
#             print(x)
#             if x not in name_list:
#                 del New_Data[0]["name"]
#         return New_Data
#             # print"%s : %s" %(temp_key,temp_value)
#             # list_all_dict(temp_value) #自我调用实现无限遍历
#
# New_Datasets_path = '/Users/zhangyujuan/graduation/New_Datasets.json'
# new_data_path = '/Users/zhangyujuan/graduation/new_data.json'

# New_Data = read_json(New_Datasets_path)
# print(len(New_Data))



# print(len(name_list))
# print(name_list)

# new_data = read_json(new_data_path)
# print(len(new_data))
# new_data2 = list_all_dict(new_data[0])
# for key, value in new_data[0]:
#     print(value)

# new_data_dict = new_data[0]
#
# for i in New_Data:  # 遍历所有名字
#     # 增加 new name 以及 ingredients
#     if i["name"] in new_data:
#         new_data[i["name"]]["ingredients"]=i["ingredients"]
#         new_data[i["name"]]["new_name"] = i["new_name"]
#
# print(new_data)
# now_test_path = '/Users/zhangyujuan/graduation/wawa.json'
# write_json(new_data_path,new_data)

# print("="*100)
# New_Data = read_json(New_Datasets_path)  # 1391
# print(len(New_Data))

# print("-"*100)
# new_data = read_json(new_data_path) # 1744
# print(len(new_data))
#
# print("+"*100)
# final_file_path = '/Users/zhangyujuan/graduation/ingredients.json'
# wawa = read_json(final_file_path)  # 1377
# print(len(wawa))

# =================================== 增加details信息 ======================================================
import json

# def write_json(file_path,data):
#     with open(file_path, 'w') as f:    #在代码当前目录生成一个data.json的文件
#         json.dump(data, f)
#
# def read_json(file_path):
#     with open(file_path, 'r') as f:  # 读取当前目录的json文件并解码成python数据
#         data = json.load(f)
#         print(data)
#         return data
#
# def list_all_dict(dict_a):
#     if isinstance(dict_a,dict) : #使用isinstance检测数据类型
#         for x in (dict_a):
#             print(x)
#
# New_Datasets_2_path = '/Users/zhangyujuan/graduation/New_Datasets_2.json'
# details_path = '/Users/zhangyujuan/graduation/details.json'
#
# New_Datasets_2 = read_json(New_Datasets_2_path)
# details = read_json(details_path)
#
# for x in details:
#     if x in New_Datasets_2:
#         New_Datasets_2[x]["details"] = details[x]["details"]
#
# print(New_Datasets_2)
# save_file = 'c.json'
# write_json(save_file,New_Datasets_2)

# =================================== 增加class信息 ======================================================
import json
import copy

def write_json(file_path,data):
    with open(file_path, 'w') as f:    #在代码当前目录生成一个data.json的文件
        json.dump(data, f)

def read_json(file_path):
    with open(file_path, 'r') as f:  # 读取当前目录的json文件并解码成python数据
        data = json.load(f)
        print(data)
        return data

file_path = "/Users/zhangyujuan/graduation/final.json"
data = read_json(file_path)
new = copy.deepcopy(data)
for i in data:
    if data[i]["category"] == "Face Serums":
        new[i]["class"] = "Serum"
    elif data[i]["category"] == "Moisturizers" or data[i]["category"] == "Moisturizer & Treatments":
        new[i]["class"] = "Cream"
    elif data[i]["category"] == "Eye Creams & Treatments" or data[i]["category"] == "Eye Cream":
        new[i]["class"] = "EyeCream"
    elif data[i]["category"] == "Face Sunscreen" or data[i]["category"] == "Sunscreen":
        new[i]["class"] = "Sunscreen"
    elif data[i]["category"] == "Blemish & Acne Treatments":
        new[i]["class"] = "Acne"
    elif data[i]["category"] == "Face Oils":
        new[i]["class"] = "Oils"
    elif data[i]["category"] == "Face Wash & Cleansers":
        new[i]["class"] = "Cleanser"
    elif data[i]["category"] == "Lotions & Oils":
        del new[i]
    elif data[i]["category"] == "Night Creams":
        new[i]["class"] = "NightCream"
    elif data[i]["category"] == "Toners" or data[i]["category"] == "Mists & Essences":
        new[i]["class"] = "Toner"
    else:
        pass
print(new)
tt = 'd.json'
write_json(tt,new)