import math
from operator import itemgetter

def UserSimilarity1(train):
    if False:
        i = 10
        return i + 15
    W = dict()
    for u in train.keys():
        for v in train.keys():
            if u == v:
                continue
            W[u][v] = len(train[u] & train[v])
            W[u][v] /= math.sqrt(len(train[u]) * len(train[v]) * 1.0)
    return W

def UserSimilarity2(train):
    if False:
        print('Hello World!')
    item_users = dict()
    for (u, items) in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)
    C = dict()
    N = dict()
    for (i, users) in item_users.items():
        for u in users:
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                C[u][v] += 1
    W = dict()
    for (u, related_users) in C.items():
        for (v, cuv) in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W

def UserSimilarity3(train):
    if False:
        i = 10
        return i + 15
    item_users = dict()
    for (u, items) in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)
    C = dict()
    N = dict()
    for (i, users) in item_users.items():
        for u in users:
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                C[u][v] += 1 / math.log(1 + len(users))
    W = dict()
    for (u, related_users) in C.items():
        for (v, cuv) in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W

def Recommend(user, train, W):
    if False:
        for i in range(10):
            print('nop')
    rank = dict()
    interacted_items = train[user]
    for (v, wuv) in sorted(W[u].items, key=itemgetter(1), reverse=True)[0:K]:
        for (i, rvi) in train[v].items:
            if i in interacted_items:
                continue
            rank[i] += wuv * rvi
    return rank