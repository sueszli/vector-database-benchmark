import math
import random

def SplitData(data, M, k, seed):
    if False:
        return 10
    test = []
    train = []
    random.seed(seed)
    for (user, item) in data:
        if random.randint(0, M) == k:
            test.append([user, item])
        else:
            train.append([user, item])
    return (train, test)

def Precision(train, test, N):
    if False:
        while True:
            i = 10
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user, N)
        for (item, pui) in rank:
            if item in tu:
                hit += 1
        all += N
    return hit / (all * 1.0)

def Recall(train, test, N):
    if False:
        print('Hello World!')
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user, N)
        for (item, pui) in rank:
            if item in tu:
                hit += 1
        all += len(tu)
    return hit / (all * 1.0)

def Coverage(train, test, N):
    if False:
        while True:
            i = 10
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        rank = GetRecommendation(user, N)
        for (item, pui) in rank:
            recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)

def Popularity(train, test, N):
    if False:
        print('Hello World!')
    item_popularity = dict()
    for (user, items) in train.items():
        for item in items.keys():
            if item not in item_popularity:
                item_popularity[item] = 0
                item_popularity[item] += 1
    ret = 0
    n = 0
    for user in train.keys():
        rank = GetRecommendation(user, N)
        for (item, pui) in rank:
            ret += math.log(1 + item_popularity[item])
            n += 1
    ret /= n * 1.0
    return ret