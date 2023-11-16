#!/usr/bin/env python
# -*- coding:utf-8 -*-

import json
import re
from py2neo import *
from re import search

def write_json(file_path,data):
    with open(file_path, 'w') as f:    #在代码当前目录生成一个data.json的文件
        json.dump(data, f)

def read_json(file_path):
    with open(file_path, 'r') as f:  # 读取当前目录的json文件并解码成python数据
        data = json.load(f)
        print(data)
        return data


graph = Graph("bolt://localhost:7687/neo4j", username="neo4j", password='123456')
matcher = NodeMatcher(graph)


# =========================================== 1. 转换成Cypher语言 创建1362个结点 ===========================================
# dataset_path = '/Users/zhangyujuan/graduation/finally.json'
# test_path = '/Users/zhangyujuan/graduation/test2.json'
# f = open('test.txt','a')
# data = read_json(dataset_path)
# for x in data:
#
#     name_index = str(x).replace(" ",'').replace("-",'').replace("+","")
#     pattern = "\w+"
#     b = re.findall(pattern, name_index)[0]
#     line = "CREATE (" + name_index + ":" + data[x]["class"] + "{" + "name:" + '"' + str(x) + '"' + "," + "brand:" + '"' \
#            + data[x]["brand"] + '"' + "," + "rating:" + str(data[x]["rating"]) + "," + "love:" + str(data[x]["love"]) + \
#            "," + "price:" + str(data[x]["price"]) + "," + "URL:" + '"' + data[x]["URL"] + '"' + "," + "how_to_use:" + '"' + data[x]["how_to_use"] \
#            + '"' + "," + "new_name:" + '"' + data[x]["new_name"] + '"' + "})"
#     print(line)
#     f.write('\n')
#     f.write(line)
# f.close()

# =========================================== 2. 创建成份结点 ===========================================================

# dataset_path = '/Users/zhangyujuan/graduation/ingredients.json'
# test_path = '/Users/zhangyujuan/graduation/test2.json'
# f = open('2-ingredients-cypher.txt','a')
# data = read_json(dataset_path)
# for x in data:
#
#     name_index = str(x).replace(" ",'').replace("-",'').replace("+","").replace(".","").replace("%","")
#     pattern = "[A-Za-z]"
#     b = re.findall(pattern, name_index)[0]
#     line = "CREATE (" + name_index + ":" +"Ingredients" + "{" + "name:" + '"' + str(x)+'"' + ','+ "chinese:" + '"' + data[x]["chinese"]\
#            + '"' + ',' + "effcet:" + '"' + data[x]["effect"] + '"' + ',' + "acne:"+'"' + str(data[x]["acne"])+'"'+','+"irritation:" \
#            + '"' + data[x]["irritation"] + '"' + ',' + "safety:" + '"' + data[x]["safety"] + '"' + "})"
#     print(line)
#     f.write('\n')
#     f.write(line)
# f.close()


# =========================================== 2. 创建成份结点 ===========================================================
# from py2neo import *
# graph = Graph("bolt://localhost:7687/neo4j", username="neo4j", password='123456')
# matcher = NodeMatcher(graph)
#
# dataset_path = '/Users/zhangyujuan/graduation/finally.json'
#
# # dataset_path = '/Users/zhangyujuan/graduation/test3.json'
# data = read_json(dataset_path)
# for x in data:
#     ingre_lst = data[x]["ingredients"]
#     print(len(ingre_lst))
#     a = graph.nodes.match(data[x]["class"], name=x).first()
#     for i in ingre_lst:
#         b = graph.nodes.match("Ingredients", name=i).first()
#         if a and b:
#             r = Relationship(a, "hasIngredient", b)
#             graph.create(r)
#
# # # 建立已有结点的关系
# # a_have=graph.nodes.match("Person",name="Alice").first()
# # b_have=graph.nodes.match("Person",name="Bob").first()
# # rel_a=Relationship(a_have,"likes",b_have)
# # graph.create(rel_a)
# #
# # # 查找目前所有的关系
# # find_re = RelationshipMatcher(graph)
# # find_r_list = list(find_re.match())
# # for i in find_r_list:
# #     print(i)
# # # 删除目前所有的关系
# #
# # i = 0
# # while i<30:
# #     relationship = graph.match_one(r_type='hasIngredient')
# #
# #     graph.delete(relationship)
# #     i+=1
# =========================================== 3. 创建肤质关系 ===========================================================
# # "details": ["Normal", "Dry", "Combination", "Oily"]
dataset_path = '/Users/zhangyujuan/graduation/finally.json'

def made_relation_skintype(data):
    for x in data:
        details_lst = data[x]["details"]
        print(len(details_lst))
        a = graph.nodes.match(data[x]["class"], name=x).first()
        for i in details_lst:
            b = graph.nodes.match("Skintype", name=i).first()
            if a and b:
                r = Relationship(a, "suitsfor", b)
                graph.create(r)


# =========================================== 4. 创建成份的作用关系 ===========================================================
# # 美白：Whitening
# # 保湿：Moisturizing
# # 除皱：Anti-wrinkle
# # 滋润：Moisten
# # 收敛：Porerefining

#
# dataset_path = '/Users/zhangyujuan/graduation/ingredients.json'
#
# # dataset_path = '/Users/zhangyujuan/graduation/test4.json'
# data = read_json(dataset_path)
# pattern_Moisturizing =re.compile("保湿")
# pattern_Whitening =re.compile("美白")
# pattern_Moisten =re.compile("滋润")
# pattern_Anti =re.compile("抗氧化")
# pattern_wrinkle =re.compile("除皱")
# pattern_Porerefining =re.compile("收敛")
#
# pattern_flavor =re.compile("香精")
# pattern_Ethoxyquim =re.compile("防腐剂")
#
# for x in data:
#     if data[x]["effect"]:
#         if pattern_Moisturizing.search(data[x]["effect"]):
#             print(data[x]["effect"], ": ", "保湿")
#             a = graph.nodes.match("Ingredients", name=x).first()
#             b = graph.nodes.match("Function", name="Moisturizing").first()
#             if a and b:
#                 r = Relationship(a, "haseffect", b)
#                 graph.create(r)
#
#         if pattern_Whitening.search(data[x]["effect"]):
#             print(data[x]["effect"], ": ", "美白")
#             a = graph.nodes.match("Ingredients", name=x).first()
#             b = graph.nodes.match("Function", name="Whitening").first()
#             if a and b:
#                 r = Relationship(a, "haseffect", b)
#                 graph.create(r)
#
#         if pattern_Moisten.search(data[x]["effect"]):
#             print(data[x]["effect"], ": ", "滋润")
#             a = graph.nodes.match("Ingredients", name=x).first()
#             b = graph.nodes.match("Function", name="Moisten").first()
#             if a and b:
#                 r = Relationship(a, "haseffect", b)
#                 graph.create(r)
#
#         if pattern_Anti.search(data[x]["effect"]) or pattern_wrinkle.search(data[x]["effect"]):
#             print(data[x]["effect"],": ","抗氧化  除皱")
#             a = graph.nodes.match("Ingredients", name=x).first()
#             b = graph.nodes.match("Function", name="Anti-wrinkle").first()
#             if a and b:
#                 r = Relationship(a, "haseffect", b)
#                 graph.create(r)
#
#         if pattern_Porerefining.search(data[x]["effect"]):
#             print(data[x]["effect"], ": ", "收敛")
#             a = graph.nodes.match("Ingredients", name=x).first()
#             b = graph.nodes.match("Function", name="Porerefining").first()
#             if a and b:
#                 r = Relationship(a, "haseffect", b)
#                 graph.create(r)
#         if pattern_flavor.search(data[x]["effect"]):
#             print(data[x]["effect"], ": ", "收敛")
#             a = graph.nodes.match("Ingredients", name=x).first()
#             b = graph.nodes.match("Flavor", name="Porerefining").first()
#             if a and b:
#                 r = Relationship(a, "haseffect", b)
#                 graph.create(r)



# =========================================== 5. 补充肤质关系 ===========================================================

B_json_path = '/Users/zhangyujuan/graduation/B_finally.json'
# data = read_json(B_json_path)
# made_relation_skintype(data)

# =========================================== 补充节点的属性 ===================================================

# =========================================== 6. 补充成份之间互斥的关系，依据属性值查找节点 ===========================================================

# a = list(matcher.match("Ingredients").where("_.chinese=~'(.*)烟酰胺(.*)'"))
# print(len(a))
# for i in a:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)
#
# b = list(matcher.match("Ingredients").where("_.chinese=~'(.*)维生素C(.*)'"))
# print(len(b))
# for i in b:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)
#
# c = list(matcher.match("Ingredients").where("_.chinese=~'(.*)视黄醇(.*)'"))
# print(len(c))
# for i in c:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)
#
# d = list(matcher.match("Ingredients").where("_.chinese=~'(.*)果酸(.*)'"))
# print(len(d))
# for i in d:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)
#
# e = list(matcher.match("Ingredients").where("_.chinese=~'(.*)酒精(.*)'"))
# print(len(e))
# for i in e:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)

# f = list(matcher.match("Ingredients").where("_.chinese=~'(.*)蛋白质(.*)'"))
# print(len(f))
# for i in f:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)
#
# g = list(matcher.match("Ingredients").where("_.chinese=~'(.*)卡波姆(.*)'"))
# print(len(g))
# for i in g:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)
#
# h = list(matcher.match("Ingredients").where("_.chinese=~'(.*)二氧化钛(.*)'"))
# print(len(h))
# for i in h:
#     print(i["chinese"]," ==== ",i["name"])
#
# k = list(matcher.match("Ingredients").where("_.chinese=~'(.*)水杨酸(.*)'"))
# print(len(k))
# for i in k:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)

# l = list(matcher.match("Ingredients").where("_.chinese=~'(.*)传明酸(.*)'"))
# print(len(l))
# for i in l:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)
#
# p = list(matcher.match("Ingredients").where("_.chinese=~'(.*)透明质酸(.*)'"))
# print(len(p))
# for i in p:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)
#
# q = list(matcher.match("Ingredients").where("_.chinese=~'(.*)甲基异噻唑啉酮(.*)'"))
# print(len(q))
# for i in q:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)
#
# r = list(matcher.match("Ingredients").where("_.chinese=~'(.*)神经酰胺(.*)'"))
# print(len(r))
# for i in r:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)
#
# s = list(matcher.match("Ingredients").where("_.chinese=~'(.*)角鲨烷(.*)'"))
# print(len(s))
# for i in s:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)

# 视黄醇  维生素C： c and b
# 视黄醇，果酸：c and d
# 酒精，蛋白质：e and f
# 卡波姆，二氧化钛：g and h
# 烟酰胺，水杨酸：a and k
# 烟酰胺，果酸：a and d
# 烟酰胺，维生素【Vitamin C】：
# 水杨酸，果酸：k and d

# for i in k:
#     for j in d:
#         m = graph.nodes.match("Ingredients", name=i["name"]).first()
#         n = graph.nodes.match("Ingredients", name=j["name"]).first()
#         if m and n:
#             r = Relationship(m, "in conflict with", n)
#             graph.create(r)

# =========================================== 7. 补充成份之间互补的关系   ==================================
# 视黄醇，烟酰胺：a and c
# 透明质酸，神经酰胺，角鲨烷：p ，r and s
# for i in r:
#     for j in s:
#         m = graph.nodes.match("Ingredients", name=i["name"]).first()
#         n = graph.nodes.match("Ingredients", name=j["name"]).first()
#         if m and n:
#             r = Relationship(m, "work well with", n)
#             graph.create(r)

# =========================================== 8. 敏感肌添加防腐剂   ==================================

# a = list(matcher.match("Ingredients").where("_.chinese=~'(.*)羟苯丁酯(.*)'"))
# print(len(a))
# for i in a:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)
#
# b = list(matcher.match("Ingredients").where("_.chinese=~'(.*)椰油酰胺(.*)'"))
# print(len(b))
# for i in b:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)

# c = list(matcher.match("Ingredients").where("_.chinese=~'(.*)辛甘醇(.*)'"))
# print(len(c))
# for i in c:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)

# d = list(matcher.match("Ingredients").where("_.chinese=~'(.*)乙内酰脲(.*)'"))
# print(len(d))
# for i in d:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)

# e = list(matcher.match("Ingredients").where("_.chinese=~'(.*)咪唑烷基脲(.*)'"))
# print(len(e))
# for i in e:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)

# f = list(matcher.match("Ingredients").where("_.chinese=~'(.*)碘丙炔醇丁基氨甲酸酯(.*)'"))
# print(len(f))
# for i in f:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)

# g = list(matcher.match("Ingredients").where("_.chinese=~'(.*)甲基异噻唑啉酮(.*)'"))
# print(len(g))
# for i in g:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)

# h = list(matcher.match("Ingredients").where("_.chinese=~'(.*)甲基氯异噻唑啉酮(.*)'"))
# print(len(h))
# for i in h:
#     print(i["chinese"]," ==== ",i["name"])

# k = list(matcher.match("Ingredients").where("_.chinese=~'(.*)氢化蓖麻油(.*)'"))
# print(len(k))
# for i in k:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)

# l = list(matcher.match("Ingredients").where("_.chinese=~'(.*)羟苯丙酯(.*)'"))
# print(len(l))
# for i in l:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)

# p = list(matcher.match("Ingredients").where("_.chinese=~'(.*)月桂醇硫酸酯钠(.*)'"))
# print(len(p))
# for i in p:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)

# q = list(matcher.match("Ingredients").where("_.chinese=~'(.*)三氯生(.*)'"))
# print(len(q))
# for i in q:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)
#
# r = list(matcher.match("Ingredients").where("_.chinese=~'(.*)三氯卡班(.*)'"))
# print(len(r))
# for i in r:
#     print(i["chinese"]," ==== ",i["name"])
# print("="*10)

# sensitive_skin = graph.nodes.match("Skintype", name="Sensitive").first()
# for i in l:
#     n = graph.nodes.match("Ingredients", name=i["name"]).first()
#     if sensitive_skin and n:
#         r = Relationship(sensitive_skin, "in conflict with", n)
#         graph.create(r)


