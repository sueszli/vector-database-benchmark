from sacred import Ingredient, Experiment
data_ingredient = Ingredient('dataset')

@data_ingredient.config
def cfg1():
    if False:
        for i in range(10):
            print('nop')
    filename = 'my_dataset.npy'
    normalize = True

@data_ingredient.capture
def load_data(filename, normalize):
    if False:
        return 10
    print("loading dataset from '{}'".format(filename))
    if normalize:
        print('normalizing dataset')
        return 1
    return 42

@data_ingredient.command
def stats(filename, foo=12):
    if False:
        i = 10
        return i + 15
    print('Statistics for dataset "{}":'.format(filename))
    print('mean = 42.23')
    print('foo=', foo)

@data_ingredient.config
def cfg2():
    if False:
        print('Hello World!')
    filename = 'foo.npy'
ex = Experiment('my_experiment', ingredients=[data_ingredient])

@ex.config
def cfg3():
    if False:
        i = 10
        return i + 15
    a = 12
    b = 42

@ex.named_config
def fbb():
    if False:
        for i in range(10):
            print('nop')
    a = 22
    dataset = {'filename': 'AwwwJiss.py'}

@ex.automain
def run():
    if False:
        return 10
    data = load_data()
    print('data={}'.format(data))