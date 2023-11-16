#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'nastra'

from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import sys
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import datasets
from time import time


english_stemmer = SnowballStemmer('english')


class StemmedTfIdfCountVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfIdfCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


def distance_raw(v1, v2):
    delta = v1 - v2
    return np.linalg.norm(delta.toarray())


def distance_normalized(v1, v2):
    v1_normalized = v1 / np.linalg.norm(v1.toarray())
    v2_normalized = v2 / np.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return np.linalg.norm(delta.toarray())


def load_data_from_dir(directory, delimiter):
    files = [open(os.path.join(directory, f)).read() for f in os.listdir(directory)]
    out = []
    for f in files:
        out.extend(f.split(delimiter))
    return out


def get_similar_posts(X, post, posts):
    import sys

    shortest_dist = sys.maxint
    num_samples, num_features = X.shape
    post_vectorized = vectorizer.transform([post])
    best_post = None
    best_post_index = None

    for i in range(0, num_samples):
        current_post = posts[i]
        if current_post == post:
            continue
        curr_post_vectorized = X.getrow(i)
        dist = distance_normalized(curr_post_vectorized, post_vectorized)
        print "Post %i: '%s' with distance= %.2f" % (i, current_post, dist)
        if dist < shortest_dist:
            shortest_dist = dist
            best_post_index = i
            best_post = current_post

    if best_post_index is not None:
        return X.getrow(best_post_index), best_post, shortest_dist
    return None, None, None


def bench_k_means(km, name, data):
    print(120 * '=')
    t0 = time()
    km.fit(data)
    print "Algorithm -- Time -- Homogeneity -- Completeness -- V-Measure -- Adjusted Rand Index -- Adjusted Mutual Info -- Silhouette Coefficient"
    print(
        '% 9s    %.2fs   %.3f           %.3f            %.3f         %.3f                   %.3f                    %.3f'
        % (name, (time() - t0),
           metrics.homogeneity_score(labels, km.labels_),
           metrics.completeness_score(labels, km.labels_),
           metrics.v_measure_score(labels, km.labels_),
           metrics.adjusted_rand_score(labels, km.labels_),
           metrics.adjusted_mutual_info_score(labels, km.labels_),
           metrics.silhouette_score(data, km.labels_, metric='euclidean')))
    print(120 * '=')


def determine_best_cluster_size(min_clusters, max_clusters, data):
    assert min_clusters < max_clusters
    best_score = -1
    cluster_size = max_clusters
    for i in range(min_clusters, max_clusters):
        km = KMeans(init='k-means++', n_clusters=i, n_init=10)
        km.fit(data)
        score = metrics.homogeneity_score(labels, km.labels_)
        if score > best_score:
            best_score = score
            cluster_size = i
            print("Current best Homogeneity Score (%.2f) for cluster size %i" % (best_score, cluster_size))

    return cluster_size


def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))


def show_similar_posts():
    # search for similar posts in the same category
    similar_indices = (kmeans.labels_ == new_post_label).nonzero()[0]

    similar = []
    for i in similar_indices:
        dist = np.linalg.norm((new_post_vec - X_train[i]).toarray())
        similar.append((dist, train_data.data[i]))

    similar = sorted(similar)

    show_at_1 = similar[0]
    show_at_2 = similar[len(similar) / 2]
    show_at_3 = similar[-1]

    category = train_data.target_names
    print("Showing similar posts from category: '" + str(new_post_label) + "' for post: '" + str(new_post) + "'")
    print(120 * '=')
    print(show_at_1)
    print(120 * '-')
    print(show_at_2)
    print(120 * '-')
    print(show_at_3)
    print(120 * '=')


# the following things happen here:
# 1. we tokenize the posts
# 2. we throw away words that occur too often to be of any help by calculating tf–idf values
# 3. we throw away words that occur so seldom that there is only a small chance that they occur in future posts
# 4. we count the remaining words in the posts
# vectorizer = StemmedTfIdfCountVectorizer(min_df=1, stop_words='english', decode_error='ignore')
#posts = load_data_from_dir("Building_ML_Systems_with_Python/chapter_03_Codes/data/toy", "\n")
#X_train = vectorizer.fit_transform(posts)
#n_samples, n_features = X_train.shape
#print n_samples
#post = "how does machine learning work?"
#post_vec, found_post, distance = get_similar_posts(X_train, post, posts)
#
#print "\n"
#print "The most similar post to '%s' is: '%s' with distance= %.2f" % (post, found_post, distance)

MLCOMP_DIR = "Building_ML_Systems_with_Python/chapter_03_Codes/data"
categories = [
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
train_data = datasets.load_mlcomp("20news-18828", "train",
                                  mlcomp_root=MLCOMP_DIR,
                                  categories=categories)
test_data = datasets.load_mlcomp("20news-18828", "test",
                                 mlcomp_root=MLCOMP_DIR,
                                 categories=categories)
print("Number of training data posts:", len(train_data.filenames))
print("Number of test data posts:", len(test_data.filenames))

vectorizer = StemmedTfIdfCountVectorizer(min_df=10, max_df=0.5, stop_words='english', decode_error='ignore')

X_train = vectorizer.fit_transform(train_data.data)
X_test = vectorizer.transform(test_data.data)

num_train_samples, num_train_features = X_train.shape
num_test_samples, num_test_features = X_test.shape
labels = train_data.target

print("# training samples: %d, # training features: %d" % (num_train_samples, num_train_features))
print("# test samples: %d, # test features: %d" % (num_test_samples, num_test_features))

n_clusters = 46
#best_cluster_size = determine_best_cluster_size(30, 50, X_train)
#print("Best cluster size for KMeans++ would be: " + str(best_cluster_size))

kmeans = KMeans(init='random', n_clusters=n_clusters, n_init=10)
bench_k_means(kmeans, name="random", data=X_train)

## in this case the seeding of the centers is deterministic, hence we run the
## kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_clusters).fit(X_train.toarray())
kmeans = KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1)
bench_k_means(kmeans, name="PCA-based", data=X_train)

kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
bench_k_means(kmeans, name="k-means++", data=X_train)

new_post = \
    """Disk drive problems. Hi, I have a problem with my hard disk.
After 1 year it is working only sporadically now.
I tried to format it, but now it doesn't boot any more.
Any ideas? Thanks.
"""
new_post_vec = vectorizer.transform([new_post])
prediction = kmeans.predict(new_post_vec)
print prediction
new_post_label = prediction[0]

show_similar_posts()


