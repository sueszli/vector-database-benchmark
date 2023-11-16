"""
.. _attributes:

User Attributes
===============

This feature is to annotate experiments with user-defined attributes.
"""
import sklearn.datasets
import sklearn.model_selection
import sklearn.svm
import optuna
study = optuna.create_study(storage='sqlite:///example.db')
study.set_user_attr('contributors', ['Akiba', 'Sano'])
study.set_user_attr('dataset', 'MNIST')
study.user_attrs
study_summaries = optuna.get_all_study_summaries('sqlite:///example.db')
study_summaries[0].user_attrs

def objective(trial):
    if False:
        for i in range(10):
            print('nop')
    iris = sklearn.datasets.load_iris()
    (x, y) = (iris.data, iris.target)
    svc_c = trial.suggest_float('svc_c', 1e-10, 10000000000.0, log=True)
    clf = sklearn.svm.SVC(C=svc_c)
    accuracy = sklearn.model_selection.cross_val_score(clf, x, y).mean()
    trial.set_user_attr('accuracy', accuracy)
    return 1.0 - accuracy
study.optimize(objective, n_trials=1)
study.trials[0].user_attrs