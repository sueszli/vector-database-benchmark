"""
==================================================
Column Transformer with Heterogeneous Data Sources
==================================================

Datasets can often contain components that require different feature
extraction and processing pipelines. This scenario might occur when:

1. your dataset consists of heterogeneous data types (e.g. raster images and
   text captions),
2. your dataset is stored in a :class:`pandas.DataFrame` and different columns
   require different processing pipelines.

This example demonstrates how to use
:class:`~sklearn.compose.ColumnTransformer` on a dataset containing
different types of features. The choice of features is not particularly
helpful, but serves to illustrate the technique.

"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
categories = ['sci.med', 'sci.space']
(X_train, y_train) = fetch_20newsgroups(random_state=1, subset='train', categories=categories, remove=('footers', 'quotes'), return_X_y=True)
(X_test, y_test) = fetch_20newsgroups(random_state=1, subset='test', categories=categories, remove=('footers', 'quotes'), return_X_y=True)
print(X_train[0])

def subject_body_extractor(posts):
    if False:
        return 10
    features = np.empty(shape=(len(posts), 2), dtype=object)
    for (i, text) in enumerate(posts):
        (headers, _, body) = text.partition('\n\n')
        features[i, 1] = body
        prefix = 'Subject:'
        sub = ''
        for line in headers.split('\n'):
            if line.startswith(prefix):
                sub = line[len(prefix):]
                break
        features[i, 0] = sub
    return features
subject_body_transformer = FunctionTransformer(subject_body_extractor)

def text_stats(posts):
    if False:
        while True:
            i = 10
    return [{'length': len(text), 'num_sentences': text.count('.')} for text in posts]
text_stats_transformer = FunctionTransformer(text_stats)
pipeline = Pipeline([('subjectbody', subject_body_transformer), ('union', ColumnTransformer([('subject', TfidfVectorizer(min_df=50), 0), ('body_bow', Pipeline([('tfidf', TfidfVectorizer()), ('best', PCA(n_components=50, svd_solver='arpack'))]), 1), ('body_stats', Pipeline([('stats', text_stats_transformer), ('vect', DictVectorizer())]), 1)], transformer_weights={'subject': 0.8, 'body_bow': 0.5, 'body_stats': 1.0})), ('svc', LinearSVC(dual=False))], verbose=True)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print('Classification report:\n\n{}'.format(classification_report(y_test, y_pred)))