from sklearn.linear_model import LogisticRegression
import numpy as np
import random

def true_classifier(i):
    if False:
        for i in range(10):
            print('nop')
    if i >= 700:
        return 1
    return 0
x = np.array([random.randint(0, 1000) for i in range(0, 1000)])
x = x.reshape((-1, 1))
y = [true_classifier(x[i][0]) for i in range(0, 1000)]
y = np.array(y)
model = LogisticRegression(solver='liblinear')
model = model.fit(x, y)
samples = [random.randint(0, 1000) for i in range(0, 100)]
samples = np.array(samples)
samples = samples.reshape(-1, 1)
_class = model.predict(samples)
proba = model.predict_proba(samples)
num_accurate = 0
for i in range(0, 100):
    if true_classifier(samples[i]) == (_class[i] == 1):
        num_accurate = num_accurate + 1
    print('' + str(samples[i]) + ': Class ' + str(_class[i]) + ', probability ' + str(proba[i]))
print('')
print(str(num_accurate) + ' out of 100 correct.')