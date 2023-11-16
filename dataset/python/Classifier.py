import joblib
import numpy as np
import pandas as pd
from sklearn import (
    metrics,
    svm,
    tree,
    ensemble, neural_network,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from Evaluation_CMC import evaluate


def work(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    report = pd.DataFrame(metrics.classification_report(y_test, y_pred, zero_division=0, output_dict=True)).T
    #  cross val to avoid overfitting
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
    n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return name, score, report, n_scores


def train_classifier(dataset, predictions):
    X = pd.read_csv(dataset)

    # encode categorical value
    enc = LabelEncoder()
    y = enc.fit_transform(X.pop('PATIENT_NAME'))
    np.save('classes.npy', enc.classes_)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)

    models = {
        'svm': svm.SVC(),
        'dtree': tree.DecisionTreeClassifier(),
        'mlpn': neural_network.MLPClassifier(),
        'randforest': ensemble.RandomForestClassifier(),
        'adaboost': ensemble.AdaBoostClassifier(),
    }

    res = joblib.Parallel(n_jobs=len(models), verbose=2)(
        joblib.delayed(work)(name, model, X_train, y_train, X_test, y_test) for name, model in models.items()
    )

    for i in range(len(models)):
        print("Model name: ", res[i][0])
        print("Score", np.mean(res[i][3]))
        print("Report")
        print(res[i][2])

    parameters = {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                  'max_features': ['auto', 'sqrt'],
                  'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                  'min_samples_split': [2, 5, 10],
                  'min_samples_leaf': [1, 2, 4],
                  'bootstrap': [True, False]}

    clf = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=parameters, n_iter=100, cv=3,
                             verbose=2, n_jobs=-1)

    clf.fit(X_train, y_train)
    print("Score on training dataset", clf.score(X_train, y_train))
    print("Best params", clf.best_params_)
    print("Score on testing dataset", clf.score(X_test, y_test))

    joblib.dump(clf.best_estimator_, 'model.joblib', compress=3)

    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)

    # y_pred = enc.inverse_transform(y_pred)
    # y_test = enc.inverse_transform(y_test)

    new_df = pd.DataFrame(y_test, columns=['REAL'])
    new_df.insert(0, "PREDICTED", y_pred)
    new_df.insert(2, "SCORES", list(y_scores))

    new_df.to_csv(predictions, index=False)

    evaluate(predictions)
