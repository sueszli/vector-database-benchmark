#!/usr/bin/env python
# -*- coding:utf-8 -*-

import requests
import csv
from py2neo import *
import json



# 将B_result数据补充到B.json
def read_csv(file_path):
    with open(file_path) as f:
        label_file = csv.reader(f)
        data = []
        for online in label_file:
            data.append(online)
        return data

def create_csv(path,data1):
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data1)

def read_json(file_path):
    with open(file_path, 'r') as f:  # 读取当前目录的json文件并解码成python数据
        data = json.load(f)
        print(data)
        return data


def get_city_code(str_c):
    city_code_csv_file="/Users/zhangyujuan/graduation/recommendation/city_code_csv.csv"
    with open(city_code_csv_file) as csvfile:
        reader = csv.reader(csvfile)
        column = [row for row in reader]
        for i in range(len(column)):
            if str_c in column[i][1]:
                return column[i][1],column[i][0]
        else:
            return 0
# city_name, city_id = get_city_code("汕头")
def get_weather(city_name, city_id):
    ak = 'HELaVmuS91N8z1IjE9i3m2uAVFWGMVBU' #你的ak
    url = 'http://api.map.baidu.com/weather/v1/?district_id=%d&data_type=all&ak=%s'%(int(city_id),ak)
    a = requests.get(url).json()
    result = a['result']['forecasts'][0]
    rh = a['result']['now']['rh']
    text = a['result']['now']['text']
    date = result['date'] #当天日期
    text_day = result['text_day'] #白天天气
    text_night = result['text_night'] #晚上天气
    high_temp = result['high'] #当天最高温度℃
    low_temp = result['low'] #当天最低温度℃
    wc_day = result['wc_day'] #风力
    wd_day = result['wd_day'] #风
    weather={}
    weather["日期"]=date
    weather["城市名称"]=city_name
    weather["天气"]=text
    weather["白天天气"]=text_day
    weather["晚上天气"]=text_night
    weather["最高温度"]=high_temp
    weather["最低温度"] =low_temp
    weather["湿度"] =rh
    weather["风力"] =wc_day
    weather["风"] = wd_day
    if weather:
        return weather
    else:
        return 0



dict_skin_type = {"1":"Dry","2":"Oily","3":"Combination","4":"Normal","5":"Sensitive"}
dict_type_eng = {"1":"Cleanser","2":"Toner","3":"Cream","4":"Serum","5":"EyeCream","6":"Sunscreen"}
# skin_type 不包含祛痘。
# Mositen滋润
dict_func = {"1":"Moisturizing","2":"Whitening","3":"Anti-wrinkle","4":"Porerefining","5":"Acne","6":"Moisten"}

def skintype_input():
    print("=-*-="*10)
    input_num = input("请输入肤质序号：\n"
                     "    1. 干性 \n"
                     "    2. 油性 \n"
                     "    3. 混合 \n"
                     "    4. 正常 \n")
    if int(input_num)<5 and int(input_num)>0:
        return dict_skin_type[(input_num)]
    return skintype_input()


def sensitive_input():
    print("=-*-="*10)
    input_num = input("是否为敏感肌：\n"
                     "    1. 是 \n"
                     "    2. 否 \n")
    if int(input_num)<3 and int(input_num)>0:
        return int(input_num)
    return sensitive_input()


def service_input():
    print("=-*-="*10)
    input_num = input("请选择待服务内容：\n"
                     "    1. 查推荐 \n"
                     "    2. 查搭配 \n")
    if int(input_num)<3 and int(input_num)>0:
        return int(input_num)
    return service_input()

def type_re_input():
    print("=-*-="*10)
    input_list = list(map(int,input("请选择待推荐的种类序号，按逗号分割，（如 2,3,4）：：\n"
                     "    0. 推荐方案： \n"
                     "        方案一：1,2,6 \n"
                     "        方案二：1,2,3,6 \n"
                     "        方案三：1,2,3,4,5,6 \n"
                     "    1. 洗面奶 \n"
                     "    2. 护肤水 \n"
                     "    3. 乳液/面霜 \n"
                     "    4. 精华 \n"
                     "    5. 眼霜 \n"
                     "    6. 防晒 \n"
                      ).strip().split(",")))
    if input_list:
        return input_list
    return type_re_input()

def type_match_input():
    print("=-*-="*10)
    input_list = list(map(int,input("请选择待搭配的种类序号，按逗号分割，（如 1,2,3）：\n"
                     "    1. 洗面奶 \n"
                     "    2. 护肤水 \n"
                     "    3. 乳液/面霜 \n"
                     "    4. 精华 \n"
                     "    5. 眼霜 \n"
                     "    6. 防晒 \n"
                      ).strip().split(",")))
    if input_list:
        return input_list
    return type_match_input()

def city_input():
    print("=-*-="*10)
    input_string = input("请输入所在城市：\n")
    if input_string:
        return input_string
    return city_input()

def function_input():
    print("=-*-="*10)
    input_list = list(map(int,input("请输入功能需求：\n"
                     "    1. 保湿（默认） \n"
                     "    2. 美白 \n"
                     "    3. 去皱 \n"
                     "    4. 收敛 \n"
                     "    5. 祛痘 \n"
                      )))
    if input_list:
        return input_list[0]
    return function_input()

def price_input(type_list):
    print("=-*-="*10)
    input_list = []
    print("请依次输入价格需求：\n")
    for i in type_list:
        j = str(i)
        words = dict_type[j] + ' 价格 ¥ ：'
        line = float(input(words))
        input_list.append(line)

    if input_list:
        return input_list
    return price_input(type_list)

def match_name_input(type_list):
    print("=-*-="*10)
    input_list = []
    print("请依次输入待搭配产品名称：\n")
    for i in type_list:
        j = str(i)
        words = dict_type[j] + ' 名称 ：'
        line = input(words)
        input_list.append(line)
    if input_list:
        return input_list
    return match_name_input(type_list)


# city = "汕头"
# name,id = get_city_code(city)
# weather = get_weather(name,id)
# print(weather)

dict_type = {}
dict_type["0"] = "待推荐"
dict_type["1"] = "洗面奶"
dict_type["2"] = "护肤水"
dict_type["3"] = "乳液/面霜"
dict_type["4"] = "精华"
dict_type["5"] = "眼霜"
dict_type["6"] = "防晒"

# dict_match_type = {}
# dict_match_type["1"] = "洗面奶"
# dict_match_type["2"] = "护肤水"
# dict_match_type["3"] = "乳液/面霜"
# dict_match_type["4"] = "精华"
# dict_match_type["5"] = "眼霜"
# dict_match_type["6"] = "防晒"



def user_input():
    skin_type = skintype_input()
    sensitive_num = sensitive_input()
    service_num = service_input()
    if service_num==1:
        # 推荐种类
        re_list = type_re_input()
        re_list_name = []
        # 换成英文列表
        for i in re_list:
            re_list_name.append(dict_type_eng[str(i)])
        city = city_input()
        function_num = function_input()
        if re_list[0]==0: # 如果包含默认推荐，那就依据所有的价格，推荐三种方案
            price_list = price_input([1,2,3,4,5,6])
        else:
            price_list = price_input(re_list)
        # print("skin_num : ",skin_num)
        # print("sensitive_num : ",sensitive_num)
        print("service_num : ",service_num)
        print("re_list : ",re_list)
        print("city: ", city)
        print("function_num : ", function_num)
        print("price_list : ", price_list)
    else:
        match_list = type_match_input()
        match_name_list = match_name_input(match_list)
        # print("skin_num : ",skin_num)
        # print("sensitive_num : ",sensitive_num)
        print("service_num : ",service_num)
        print("match_list : ",match_list)
        print("match_name_list : ", match_name_list)


#user_input()


graph = Graph("bolt://localhost:7687/neo4j", username="neo4j", password='123456')
matcher = NodeMatcher(graph)
matcher_relationship = RelationshipMatcher(graph)


# 按照价格、种类、肤质，在每个种类中，筛选出符合用户预期的产品
# 返回值：result = {"name":price}
# def search_single(single,single_price,skin_type):
#     up_price = single_price+5
#     down_price = single_price-5
#     single_list_1 = list(matcher.match(single).where("_.price <= "+str(up_price)).order_by("_.price"))
#     single_list_2 = list(matcher.match(single).where("_.price >= "+str(down_price)).order_by("_.price"))
#     # single_list = list(matcher.match(single, price__gt=1))
#                        # and("_.price >= "+str(down_price)))
#     single_list = list(set(single_list_1) & set(single_list_2))
#     # for i in single_list:
#     #     print(i["price"],i["name"])
#     print("筛选之前的长度： ", len(single_list))
#
#     result = {}
#     if single_list:
#         for i in single_list:
#             findnode = matcher.match(single,name=i["name"]).first()
#             findskintype = matcher.match("Skintype", name=skin_type).first()
#             a = list(graph.match((findnode,findskintype), r_type='suitsfor'))
#             if a:
#                 result[i["name"]] = i["price"]
#                 # print(i["price"], i["name"])
#         print("筛选之后的长度： ",len(result))
#
#         # 检测是否有环，推理是否存在不合理的场景
#         b = list(graph.run('match (a:'+ single + ')-[r:hasIngredient]->(x)<-[rr:`in conflict with`]-(b:Skintype{name:"'+
#                       skin_type+'"})<-[rrr:suitsfor]-(a) return a.name'))
#         if b:
#             for i in b:
#                 if i["a.name"] in result:
#                     del result[(i["a.name"])]
#         print("二次筛选之后的长度： ",len(result))
#         # for i in result:
#         #     print(result[i],i)
#         return result
#     else:
#         return 0

def search_single(single,single_price,skin_type,sensitive):
    up_price = single_price+5
    down_price = single_price-5
    single_list_1 = list(matcher.match(single).where("_.price <= "+str(up_price)).order_by("_.price"))
    single_list_2 = list(matcher.match(single).where("_.price >= "+str(down_price)).order_by("_.price"))
    single_list = list(set(single_list_1) & set(single_list_2))
    # for i in single_list:
    #     print(i["price"],i["name"])
    print("1、根据【价格、种类】筛选后的长度： ", len(single_list))  # 按价格筛选之后的结果

    result = {}
    if single_list:
        for i in single_list:
            findnode = matcher.match(single,name=i["name"]).first()
            findskintype = matcher.match("Skintype", name=skin_type).first()
            # 判断当前【节点】是否适合 当前【肤质】
            a = list(graph.match((findnode,findskintype), r_type='suitsfor'))  # 适合其他肤质的节点
            if a:  # 适合的话，再判断是否适合敏感肌：
                if sensitive==1:
                    findskintype2 = matcher.match("Skintype", name="Sensitive").first()
                    # 判断当前【节点】是否适合【敏感肌】
                    c = list(graph.match((findnode,findskintype2), r_type='suitsfor'))  # 适合敏感肤质的节点
                    if c:
                        result[i["name"]] = i["price"]
                else:  # 如果不是敏感肌，只是其他正常肌肤，直接保存该节点。
                    result[i["name"]] = i["price"]
                # print(i["price"], i["name"])
        print("2、根据【肤质】筛选后的长度： ",len(result))

        if sensitive==1:
            d = list(graph.run(
                'match (a:' + single + ')-[r:hasIngredient]->(x)<-[rr:`in conflict with`]-(b:Skintype{name:"' +
                'Sensitive' + '"})<-[rrr:suitsfor]-(a) return a.name'))
            if d:
                for i in d:
                    if i["a.name"] in result:
                        del result[(i["a.name"])]
            print("3、根据【敏感肌】环路判断后的长度： ", len(result))

        b = list(graph.run('match (a:'+ single + ')-[r:hasIngredient]->(x)<-[rr:`in conflict with`]-(b:Skintype{name:"'+
                      skin_type+'"})<-[rrr:suitsfor]-(a) return a.name'))  # 检测是否有环，推理是否存在不合理的场景
        if b:
            for i in b:
                if i["a.name"] in result:
                    del result[(i["a.name"])]
        print("3、最终：根据【肤质环路】判断后的长度： ",len(result))
        # for i in result:
        #     print(result[i],i)
        return result
    else:
        return 0
# search_list = search_single("Toner", 36, "Oily",1)
# print(search_list)


# 每个产品之间是否可达。
def item2item(item1,type1,item2,type2) -> bool:
    # match_item2item = 'MATCH (n:Ingredients{name:"'+item1+'"}),(m:Ingredients{name:"'+item2+'"}) ' \
    #               'with n, m ' \
    #               'match (p:'+type1+')-[r:hasIngredient]->(n),(q:'+type2+')-[rr:hasIngredient]->(m) ' \
    #               'return n,m,r,rr,p,q'

    match_item2item_inconflict = 'match (p:'+type1+'{name:"'+item1+'"})-[r:hasIngredient]->(n:Ingredients),' \
                                 '(q:'+type2+'{name:"'+item2+'"})-[rr:hasIngredient]->(m:Ingredients) ' \
                                 'with n,m,r,rr,p,q match (n)-[k:`in conflict with`]-(m) ' \
                                 'return n,m,r,rr,p,q,k'

    a = list(graph.run(match_item2item_inconflict))
    if a:
        return False
    return True


re_list = [1,2,3] # 洗面奶 水 霜
func_input_number = 4

re_list_name = []
# 换成英文列表
for i in re_list:
    re_list_name.append(dict_type_eng[str(i)])
# func = dict_func[str(func_input_number)]
func = "Whitening"
price_list = [10.0,15.0,30.0]
skin_type = "Sensitive"
sensitive_BOOL = 1


#
# def item2item(item1,type1,item2,type2) -> bool:
#     if item1=="洗面奶1" and item2=="水2":
#         return False
#     elif item1=="水1" and item2=="霜1":
#         return False
#     return True

def add_function(all_list):  # 判断推荐二维列表中，是否
    result = []
    for i in all_list:
        for j in range(len(i)):
            index_type = i.index(i[j])
            string_match = 'match (n:'+re_list_name[index_type]+\
                           '{name:"'+i[j]+'"})-[p:hasIngredient]-(m:Ingredients) ' \
                                          'with n,m match (m)-[k:haseffect]-(qq:Function{name:"'+func+'"}) r' \
                                          'eturn n'
            # print(string_match)
            run_result = list(graph.run(string_match))
            if run_result:
                result.append(i)
                break
    print("通过功能筛选每一个列表后的长度 : ", len(result))
    print("通过功能筛选每一个列表 : ", result)
    return result

# 测试代码
# all_list = [['High Performance Vitamin C Facial Serum']]
# result = add_function(all_list)
# print(result)

def recall(re_list_name,price_list,skin_type,sensitive_BOOL):
    single_list = []  # 存储每一个类别的字典，每个字典里头是name和price
    for i in range(len(re_list_name)):  # 水、乳、霜
        single_dict = search_single(re_list_name[i], price_list[i], skin_type, sensitive_BOOL)
        single_list.append(single_dict)
    print("single_list : ",single_list)
    # single_list = [{"洗面奶1":10,"洗面奶2":13},{"水1":20,"水2":22,"水3":24},{"霜1":24,"霜2":22}]
    new_group = [[]]
    if len(single_list)>2:
        count = 0
        for d in single_list:
            group=new_group.copy()
            new_group=[]
            for g in group:
                for item in d:
                    length = len(group)
                    if length>1:
                        if(item2item(g[0],re_list_name[count],item,re_list_name[count])):
                            new_group.append(g+[item])
                    else:
                        new_group.append(g+[item])
            count+=1
    else:
        new_group=[]
        for i in single_list[0]:
            new_group.append([i])
    print("new_group : ", new_group)
    result = add_function(new_group)
    return result


# 测试代码
# user_input()
# result = recall(re_list_name,price_list,skin_type,sensitive_BOOL)
# result = item2item("Clarifying Lotion 4","Toner","Refreshing Cleanser","Cleanser")
# result = recall(re_list_name,price_list,skin_type,sensitive_BOOL)
# print(len(result))




# 根据搭配的种类长度，同时读多个节点的成份，查看他们的关系是否存在 "work well with"
# 并记录存在 该功能的成份

def sum_n(n):
    if n==0:
        return 0
    else:
        s=n+sum_n(n-1)
    return s

def fun_score_single_type(name,type): # 单个item 关于含有该功能成份的 分数。
    finally_file = '/Users/zhangyujuan/graduation/finally.json'
    data = read_json(finally_file)
    s = 0.0
    match_string = 'match (n:' + type + '{name:"' + name + '"})-[p:hasIngredient]-(m:Ingredients) ' \
                                                                    'with m match (m)-[r:haseffect]' \
                                                                    '-(a:Function{name:"' + func + '"}) ' \
                                                                                                   'return m.name'
    # print(match_string)
    a = list(graph.run(match_string))
    # print(a)
    for i in range(len(a)):
        # print(i,a[i],a[i]["m.name"])
        ingredient_length = len(data[name]["ingredients"])
        m_index = (data[name]["ingredients"].index(a[i]["m.name"]) )+ 1
        tmp = float((ingredient_length - m_index) / sum_n(ingredient_length))
        s = s + tmp
        # print("a[i]['m.name'] :",a[i]["m.name"])
        # print("score : ", score)
        # print("m_index : ", m_index)
        # print(ingredient_length)
    print(name, s)
    return s

def pick_from_recall(one_list, one_list_length): #  求当前搭配list的总分数
    score = 0.0

    if one_list_length == 1:
        a_type = re_list_name[0]
        score = fun_score_single_type(one_list[0],a_type)
    if one_list_length == 2:
        a_type = re_list_name[0]
        b_type = re_list_name[1]
        # print("type : ", a_type,b_type)
        # print("name: ", )
        # a = fun_score_single_type(one_list[0], a_type)
        # b = fun_score_single_type(one_list[1], b_type)
        # print("a : ", a)
        # print("b : ", b)
        # score = a+b
        score = fun_score_single_type(one_list[0], a_type) + fun_score_single_type(one_list[1], b_type)

    if one_list_length == 3:
        a_type = re_list_name[0]
        b_type = re_list_name[1]
        c_type = re_list_name[2]
        score = fun_score_single_type(one_list[0], a_type) + \
                fun_score_single_type(one_list[1], b_type) + \
                fun_score_single_type(one_list[2], c_type)
    if one_list_length == 4:
        a_type = re_list_name[0]
        b_type = re_list_name[1]
        c_type = re_list_name[2]
        d_type = re_list_name[3]
        score = fun_score_single_type(one_list[0], a_type) + \
                fun_score_single_type(one_list[1], b_type) + \
                fun_score_single_type(one_list[2], c_type) + \
                fun_score_single_type(one_list[3], d_type)
    if one_list_length == 4:
        a_type = re_list_name[0]
        b_type = re_list_name[1]
        c_type = re_list_name[2]
        d_type = re_list_name[3]
        e_type = re_list_name[4]
        score = fun_score_single_type(one_list[0], a_type) + \
                fun_score_single_type(one_list[1], b_type) + \
                fun_score_single_type(one_list[2], c_type) + \
                fun_score_single_type(one_list[3], d_type) + \
                fun_score_single_type(one_list[4], e_type)
    if one_list_length == 6:
        a_type = re_list_name[0]
        b_type = re_list_name[1]
        c_type = re_list_name[2]
        d_type = re_list_name[3]
        e_type = re_list_name[4]
        f_type = re_list_name[5]
        score = fun_score_single_type(one_list[0], a_type) + \
                fun_score_single_type(one_list[1], b_type) + \
                fun_score_single_type(one_list[2], c_type) + \
                fun_score_single_type(one_list[3], d_type) + \
                fun_score_single_type(one_list[4], e_type) + \
                fun_score_single_type(one_list[5], f_type)
    return score


# one_list = ["Truth Juice Daily Cleanser","Glycolic Acid Exfoliating Toner","High Performance Vitamin C Facial Serum"]
# score = pick_from_recall(one_list, len(one_list))
# print("final score : ", score)

# score_result_dict = {0:1.11,1:3.22,2:2.22}
# b = sorted(score_result_dict.items(), key=lambda item: item[1], reverse=True)
# print(b)

def sort_from_score(all_list):
    score_result_dict = {} # all_list下标作为 key，分数作为value
    for i in range(len(all_list)):
        score = pick_from_recall(all_list[i],len(all_list[i]))
        score_result_dict[i]=score
    # score_result_dict = {0: 1.11, 1: 0.22, 2: 2.22,3:5.55}
    sort_list = sorted(score_result_dict.items(), key=lambda item: item[1], reverse=True) # 按value排序
    print(sort_list)
    if len(sort_list)>3:
        result = sort_list[:3]
    else:
        result = sort_list
    return result

result = recall(re_list_name,price_list,skin_type,sensitive_BOOL)
final = sort_from_score(result)
print(final)