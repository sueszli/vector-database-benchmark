__author__ = 'nastra'
#
# This script trains multinomial Naive Bayes on the tweet corpus
# to find two different results:
# - How well can we distinguish positive from negative tweets?
# - How well can we detect whether a tweet contains sentiment at all?
#

from utils import load_sanders_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import precision_recall_curve, auc, f1_score
from utils import plot_pr
from sklearn.grid_search import GridSearchCV
from sentiment_analysis_tweets_example import tweak_labels, show_all_scores
import numpy as np


def create_ngram_model(params=None):
    tfidf_ngrams = TfidfVectorizer(ngram_range=(1, 3), analyzer="word", binary=False)
    clf = MultinomialNB()
    pipeline = Pipeline([('vect', tfidf_ngrams), ('clf', clf)])
    if params:
        pipeline.set_params(**params)
    return pipeline


def train_model(clf_factory, X, Y):
    # setting random state to get deterministic behavior
    cv = ShuffleSplit(n=len(X), n_iter=10, test_size=0.3, indices=True, random_state=0)

    train_errors = []
    test_errors = []

    scores = []
    precisions, recalls, thresholds = [], [], []
    precision_recall_scores = []

    for train_index, test_index in cv:
        X_train, y_train = X[train_index], Y[train_index]
        X_test, y_test = X[test_index], Y[test_index]

        clf = clf_factory
        clf.fit(X_train, y_train)

        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)

        scores.append(test_score)
        probability = clf.predict_proba(X_test)
        precision, recall, pr_thresholds = precision_recall_curve(y_test, probability[:, 1])

        precision_recall_scores.append(auc(recall, precision))
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(pr_thresholds)

    return scores, precision_recall_scores, precisions, recalls, thresholds, test_errors, train_errors


def print_and_plot_scores(scores, pr_scores, train_errors, test_errors, precisions, recalls, name="NaiveBayes ngram"):
    scores_to_sort = pr_scores
    median = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]

    plot_pr(pr_scores[median], name, "01", precisions[median], recalls[median], label=name)

    summary = (np.mean(scores), np.std(scores),
               np.mean(pr_scores), np.std(pr_scores))
    print("AVG Scores\tSTD Scores\tAVG PR Scores\tSTD PR Scores")
    print "%.3f\t\t%.3f\t\t%.3f\t\t\t%.3f\t" % summary

    avg_train_err, avg_test_err = np.mean(train_errors), np.mean(test_errors)
    print("AVG Training Error: %.3f  -- AVG Testing Error: %.3f" % (avg_train_err, avg_test_err))
    return avg_train_err, avg_test_err


def grid_search_model(clf_factory, X, y):
    cv = ShuffleSplit(n=len(X), n_iter=10, test_size=0.3, indices=True, random_state=0)

    param_grid = dict(vect__ngram_range=[(1, 1), (1, 2), (1, 3)],
                      vect__min_df=[1, 2],
                      vect__smooth_idf=[False, True],
                      vect__stop_words=[None, "english"],
                      vect__use_idf=[True, False],
                      vect__sublinear_tf=[True, False],
                      vect__binary=[True, False],
                      clf__alpha=[0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1], )

    grid_search = GridSearchCV(clf_factory(), param_grid=param_grid, cv=cv, score_func=f1_score, verbose=10)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_


def get_best_model():
    """
    we set the parameters of the pipeline components based on the suggestion of the previous grid search.

    According to the output, the following parameters lead to the best model:

    {'vect__ngram_range': (1, 2), 'vect__smooth_idf': False, 'vect__sublinear_tf': True,
    'vect__binary': False, 'vect__min_df': 1, 'vect__stop_words': None, 'vect__use_idf': False, 'clf__alpha': 0.03}
    """
    params = dict(vect__ngram_range=(1, 2),
                  vect__min_df=1,
                  vect__smooth_idf=False,
                  vect__stop_words=None,
                  vect__use_idf=False,
                  vect__sublinear_tf=True,
                  vect__binary=True,
                  clf__alpha=0.03)
    clf = create_ngram_model(params)
    return clf


def train_and_evaluate_tuned_model(X, Y, name):
    clf = get_best_model()
    scores, precision_recall_scores, precisions, recalls, thresholds, test_errors, train_errors = train_model(
        clf, X, Y)
    print_and_plot_scores(scores, precision_recall_scores, train_errors, test_errors, precisions, recalls, name)


def show_all_scores():
    X_orig, Y_orig = load_sanders_data()
    unique_classes = np.unique(Y_orig)
    for c in unique_classes:
        print("#%s tweets: %i" % (c, sum(Y_orig == c)))

    print(120 * "#")
    print "== Pos vs. neg =="
    pos_neg = np.logical_or(Y_orig == "positive", Y_orig == "negative")
    X = X_orig[pos_neg]
    Y = Y_orig[pos_neg]
    Y = tweak_labels(Y, ["positive"])

    # best_clf, best_score, best_params = grid_search_model(create_ngram_model, X, Y)
    train_and_evaluate_tuned_model(X, Y, name="pos vs neg (tuned)")
    print(120 * "#")

    print "== Pos/neg vs. irrelevant/neutral =="
    X = X_orig
    Y = tweak_labels(Y_orig, ["positive", "negative"])
    train_and_evaluate_tuned_model(X, Y, name="sentiment vs rest (tuned)")
    print(120 * "#")

    print "== Pos vs. rest =="
    X = X_orig
    Y = tweak_labels(Y_orig, ["positive"])
    train_and_evaluate_tuned_model(X, Y, name="pos vs rest (tuned)")
    print(120 * "#")

    print "== Neg vs. rest =="
    X = X_orig
    Y = tweak_labels(Y_orig, ["negative"])
    train_and_evaluate_tuned_model(X, Y, name="neg vs rest (tuned)")
    print(120 * "#")


if __name__ == "__main__":
    show_all_scores()