"""
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
"""
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
train = 'train.csv'
test = 'test.csv'
D = 2 ** 20
alpha = 0.1

def logloss(p, y):
    if False:
        while True:
            i = 10
    p = max(min(p, 1.0 - 1e-11), 1e-11)
    return -log(p) if y == 1.0 else -log(1.0 - p)

def get_x(csv_row, D):
    if False:
        return 10
    x = [0]
    for (key, value) in csv_row.items():
        index = int(value + key[1:], 16) % D
        x.append(index)
    return x

def get_p(x, w):
    if False:
        for i in range(10):
            print('nop')
    wTx = 0.0
    for i in x:
        wTx += w[i] * 1.0
    return 1.0 / (1.0 + exp(-max(min(wTx, 20.0), -20.0)))

def update_w(w, n, x, p, y):
    if False:
        while True:
            i = 10
    for i in x:
        w[i] -= (p - y) * alpha / (sqrt(n[i]) + 1.0)
        n[i] += 1.0
    return (w, n)
w = [0.0] * D
n = [0.0] * D
loss = 0.0
for (t, row) in enumerate(DictReader(open(train))):
    y = 1.0 if row['Label'] == '1' else 0.0
    del row['Label']
    del row['Id']
    x = get_x(row, D)
    p = get_p(x, w)
    loss += logloss(p, y)
    if t % 1000000 == 0 and t > 1:
        print('%s\tencountered: %d\tcurrent logloss: %f' % (datetime.now(), t, loss / t))
    (w, n) = update_w(w, n, x, p, y)
with open('submission1234.csv', 'w') as submission:
    submission.write('Id,Predicted\n')
    for (t, row) in enumerate(DictReader(open(test))):
        Id = row['Id']
        del row['Id']
        x = get_x(row, D)
        p = get_p(x, w)
        submission.write('%s,%f\n' % (Id, p))