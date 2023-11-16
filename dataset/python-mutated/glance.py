import chainer as ch
from chainer import datasets
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import numpy as np
import matplotlib
matplotlib.use('Agg')
mushroomsfile = 'mushrooms.csv'
data_array = np.genfromtxt(mushroomsfile, delimiter=',', dtype=str, skip_header=1)
for col in range(data_array.shape[1]):
    data_array[:, col] = np.unique(data_array[:, col], return_inverse=True)[1]
X = data_array[:, 1:].astype(np.float32)
Y = data_array[:, 0].astype(np.int32)[:, None]
(train, test) = datasets.split_dataset_random(datasets.TupleDataset(X, Y), int(data_array.shape[0] * 0.7))
train_iter = ch.iterators.SerialIterator(train, 100)
test_iter = ch.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

def MLP(n_units, n_out):
    if False:
        while True:
            i = 10
    layer = ch.Sequential(L.Linear(n_units), F.relu)
    model = layer.repeat(2)
    model.append(L.Linear(n_out))
    return model
model = L.Classifier(MLP(44, 1), lossfun=F.sigmoid_cross_entropy, accfun=F.binary_accuracy)
optimizer = ch.optimizers.SGD().setup(model)
updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (50, 'epoch'), out='result')
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.DumpGraph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=(20, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.run()
(x, t) = test[np.random.randint(len(test))]
predict = model.predictor(x[None]).array
predict = predict[0][0]
if predict >= 0:
    print('Predicted Poisonous, Actual ' + ['Edible', 'Poisonous'][t[0]])
else:
    print('Predicted Edible, Actual ' + ['Edible', 'Poisonous'][t[0]])