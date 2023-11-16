__author__ = 'nastra'
from load_ml100k import load_data
import numpy as np

def all_estimates(reviews, k=1):
    if False:
        print('Hello World!')
    reviews = reviews.astype(float)
    k -= 1
    (numberOfUsers, numberOfMovies) = reviews.shape
    estimates = np.zeros_like(reviews)
    for user in range(numberOfUsers):
        userReviews = np.delete(reviews, user, 0)
        userReviews -= userReviews.mean(0)
        userReviews /= userReviews.std(0) + 0.0001
        userReviews = userReviews.T.copy()
        for movie in np.where(reviews[user] > 0)[0]:
            estimates[user, movie] = nearest_neighbor_movies(userReviews, reviews, user, movie, k)
    return estimates

def nearest_neighbor_movies(userReviews, reviews, userId, movieId, k=1):
    if False:
        print('Hello World!')
    X = userReviews
    y = userReviews[movieId].copy()
    y -= y.mean()
    y /= y.std() + 1e-05
    corrs = np.dot(X, y)
    likes = corrs.argsort()
    likes = likes[::-1]
    c = 0
    pred = 3.0
    for ell in likes:
        if ell == movieId:
            continue
        if reviews[userId, ell] > 0:
            pred = reviews[userId, ell]
            if c == k:
                return pred
            c += 1
    return pred
if __name__ == '__main__':
    reviews = load_data().toarray()
    estimates = all_estimates(reviews)
    error = estimates - reviews
    error **= 2
    error = error[reviews > 0]
    print(np.sqrt(error).mean())