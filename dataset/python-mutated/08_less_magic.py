"""A standard machine learning task without much sacred magic."""
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn import svm, datasets, model_selection
ex = Experiment('svm')
ex.observers.append(FileStorageObserver('my_runs'))
ex.add_config({'C': 1.0, 'gamma': 0.7, 'kernel': 'rbf', 'seed': 42})

def get_model(C, gamma, kernel):
    if False:
        print('Hello World!')
    return svm.SVC(C=C, kernel=kernel, gamma=gamma)

@ex.main
def run(_config):
    if False:
        for i in range(10):
            print('nop')
    (X, y) = datasets.load_breast_cancer(return_X_y=True)
    (X_train, X_test, y_train, y_test) = model_selection.train_test_split(X, y, test_size=0.2)
    clf = get_model(_config['C'], _config['gamma'], _config['kernel'])
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)
if __name__ == '__main__':
    ex.run()