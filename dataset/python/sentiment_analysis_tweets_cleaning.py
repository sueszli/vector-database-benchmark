__author__ = 'nastra'

from sentiment_analysis_tweets_tuning import tweak_labels
from sentiment_analysis_tweets_tuning import train_and_evaluate_tuned_model, train_model, print_and_plot_scores
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import re
from utils import load_sanders_data, load_sent_word_net


emoticons_replacements = {
    # positive emoticons
    "&lt;3": " good ",
    ":d": " good ",
    ":dd": " good ",
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",

    # negative emoticons:
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":S": " bad ",
    ":-S": " bad ",
}

# we need to make sure that :dd is replaced before :d
emoticons_reversed = [k for (k_len, k) in reversed(sorted([(len(k), k) for k in emoticons_replacements.keys()]))]

regex_replace = {
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
}

sent_words = load_sent_word_net()


def create_ngram_model(params=None):
    def preprocessor(tweet):
        tweet = tweet.lower()
        global emoticons_reversed

        for emoticon in emoticons_reversed:
            tweet = tweet.replace(emoticon, emoticons_replacements[emoticon])

        for regex, replacement in regex_replace.iteritems():
            tweet = re.sub(regex, replacement, tweet)

        return tweet

    tfidf_ngrams = TfidfVectorizer(ngram_range=(1, 3), analyzer="word", binary=False, preprocessor=preprocessor)
    clf = MultinomialNB()
    pipeline = Pipeline([('vect', tfidf_ngrams), ('clf', clf)])
    if params:
        pipeline.set_params(**params)
    return pipeline


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

    train_and_evaluate_tuned_model(X, Y, name="pos vs neg (cleaned)")
    print(120 * "#")

    print "== Pos/neg vs. irrelevant/neutral =="
    X = X_orig
    Y = tweak_labels(Y_orig, ["positive", "negative"])
    train_and_evaluate_tuned_model(X, Y, name="sentiment vs rest (cleaned)")
    print(120 * "#")

    print "== Pos vs. rest =="
    X = X_orig
    Y = tweak_labels(Y_orig, ["positive"])
    train_and_evaluate_tuned_model(X, Y, name="pos vs rest (cleaned)")
    print(120 * "#")

    print "== Neg vs. rest =="
    X = X_orig
    Y = tweak_labels(Y_orig, ["negative"])
    train_and_evaluate_tuned_model(X, Y, name="neg vs rest (cleaned)")
    print(120 * "#")


if __name__ == "__main__":
    show_all_scores()