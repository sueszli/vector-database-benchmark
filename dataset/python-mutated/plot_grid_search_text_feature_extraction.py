"""
==========================================================
Sample pipeline for text feature extraction and evaluation
==========================================================

The dataset used in this example is :ref:`20newsgroups_dataset` which will be
automatically downloaded, cached and reused for the document classification
example.

In this example, we tune the hyperparameters of a particular classifier using a
:class:`~sklearn.model_selection.RandomizedSearchCV`. For a demo on the
performance of some other classifiers, see the
:ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`
notebook.
"""
from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'talk.religion.misc']
data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
print(f'Loading 20 newsgroups dataset for {len(data_train.target_names)} categories:')
print(data_train.target_names)
print(f'{len(data_train.data)} documents')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('vect', TfidfVectorizer()), ('clf', ComplementNB())])
pipeline
import numpy as np
parameter_grid = {'vect__max_df': (0.2, 0.4, 0.6, 0.8, 1.0), 'vect__min_df': (1, 3, 5, 10), 'vect__ngram_range': ((1, 1), (1, 2)), 'vect__norm': ('l1', 'l2'), 'clf__alpha': np.logspace(-6, 6, 13)}
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=parameter_grid, n_iter=40, random_state=0, n_jobs=2, verbose=1)
print('Performing grid search...')
print('Hyperparameters to be evaluated:')
pprint(parameter_grid)
from time import time
t0 = time()
random_search.fit(data_train.data, data_train.target)
print(f'Done in {time() - t0:.3f}s')
print('Best parameters combination found:')
best_parameters = random_search.best_estimator_.get_params()
for param_name in sorted(parameter_grid.keys()):
    print(f'{param_name}: {best_parameters[param_name]}')
test_accuracy = random_search.score(data_test.data, data_test.target)
print(f'Accuracy of the best parameters using the inner CV of the random search: {random_search.best_score_:.3f}')
print(f'Accuracy on test set: {test_accuracy:.3f}')
import pandas as pd

def shorten_param(param_name):
    if False:
        i = 10
        return i + 15
    "Remove components' prefixes in param_name."
    if '__' in param_name:
        return param_name.rsplit('__', 1)[1]
    return param_name
cv_results = pd.DataFrame(random_search.cv_results_)
cv_results = cv_results.rename(shorten_param, axis=1)
import plotly.express as px
param_names = [shorten_param(name) for name in parameter_grid.keys()]
labels = {'mean_score_time': 'CV Score time (s)', 'mean_test_score': 'CV score (accuracy)'}
fig = px.scatter(cv_results, x='mean_score_time', y='mean_test_score', error_x='std_score_time', error_y='std_test_score', hover_data=param_names, labels=labels)
fig.update_layout(title={'text': 'trade-off between scoring time and mean test score', 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
fig
import math
column_results = param_names + ['mean_test_score', 'mean_score_time']
transform_funcs = dict.fromkeys(column_results, lambda x: x)
transform_funcs['alpha'] = math.log10
transform_funcs['norm'] = lambda x: 2 if x == 'l2' else 1
transform_funcs['ngram_range'] = lambda x: x[1]
fig = px.parallel_coordinates(cv_results[column_results].apply(transform_funcs), color='mean_test_score', color_continuous_scale=px.colors.sequential.Viridis_r, labels=labels)
fig.update_layout(title={'text': 'Parallel coordinates plot of text classifier pipeline', 'y': 0.99, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
fig