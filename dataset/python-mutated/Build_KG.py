import json
import re
from py2neo import *
from re import search

def write_json(file_path, data):
    if False:
        for i in range(10):
            print('nop')
    with open(file_path, 'w') as f:
        json.dump(data, f)

def read_json(file_path):
    if False:
        while True:
            i = 10
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(data)
        return data
graph = Graph('bolt://localhost:7687/neo4j', username='neo4j', password='123456')
matcher = NodeMatcher(graph)
dataset_path = '/Users/zhangyujuan/graduation/finally.json'

def made_relation_skintype(data):
    if False:
        for i in range(10):
            print('nop')
    for x in data:
        details_lst = data[x]['details']
        print(len(details_lst))
        a = graph.nodes.match(data[x]['class'], name=x).first()
        for i in details_lst:
            b = graph.nodes.match('Skintype', name=i).first()
            if a and b:
                r = Relationship(a, 'suitsfor', b)
                graph.create(r)
B_json_path = '/Users/zhangyujuan/graduation/B_finally.json'