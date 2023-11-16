"""
================================================
Semi-supervised Classification on a Text Dataset
================================================

In this example, semi-supervised classifiers are trained on the 20 newsgroups
dataset (which will be automatically downloaded).

You can adjust the number of categories by giving their names to the dataset
loader or setting them to `None` to get all 20 of them.

"""
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
data = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware'])
print('%d documents' % len(data.filenames))
print('%d categories' % len(data.target_names))
print()
sdg_params = dict(alpha=1e-05, penalty='l2', loss='log_loss')
vectorizer_params = dict(ngram_range=(1, 2), min_df=5, max_df=0.8)
pipeline = Pipeline([('vect', CountVectorizer(**vectorizer_params)), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(**sdg_params))])
st_pipeline = Pipeline([('vect', CountVectorizer(**vectorizer_params)), ('tfidf', TfidfTransformer()), ('clf', SelfTrainingClassifier(SGDClassifier(**sdg_params), verbose=True))])
ls_pipeline = Pipeline([('vect', CountVectorizer(**vectorizer_params)), ('tfidf', TfidfTransformer()), ('toarray', FunctionTransformer(lambda x: x.toarray())), ('clf', LabelSpreading())])

def eval_and_print_metrics(clf, X_train, y_train, X_test, y_test):
    if False:
        return 10
    print('Number of training samples:', len(X_train))
    print('Unlabeled samples in training set:', sum((1 for x in y_train if x == -1)))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Micro-averaged F1 score on test set: %0.3f' % f1_score(y_test, y_pred, average='micro'))
    print('-' * 10)
    print()
if __name__ == '__main__':
    (X, y) = (data.data, data.target)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y)
    print('Supervised SGDClassifier on 100% of the data:')
    eval_and_print_metrics(pipeline, X_train, y_train, X_test, y_test)
    y_mask = np.random.rand(len(y_train)) < 0.2
    (X_20, y_20) = map(list, zip(*((x, y) for (x, y, m) in zip(X_train, y_train, y_mask) if m)))
    print('Supervised SGDClassifier on 20% of the training data:')
    eval_and_print_metrics(pipeline, X_20, y_20, X_test, y_test)
    y_train[~y_mask] = -1
    print('SelfTrainingClassifier on 20% of the training data (rest is unlabeled):')
    eval_and_print_metrics(st_pipeline, X_train, y_train, X_test, y_test)
    print('LabelSpreading on 20% of the data (rest is unlabeled):')
    eval_and_print_metrics(ls_pipeline, X_train, y_train, X_test, y_test)