from collections import defaultdict

def _name_estimators(estimators):
    if False:
        while True:
            i = 10
    'Generate names for estimators.'
    names = [type(estimator).__name__.lower() for estimator in estimators]
    namecount = defaultdict(int)
    for (_, name) in zip(estimators, names):
        namecount[name] += 1
    for (k, v) in list(namecount.items()):
        if v == 1:
            del namecount[k]
    for i in reversed(range(len(estimators))):
        name = names[i]
        if name in namecount:
            names[i] += '-%d' % namecount[name]
            namecount[name] -= 1
    return list(zip(names, estimators))