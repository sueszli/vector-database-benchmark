import numpy as np

def get_retrieved_videos(sims, k):
    if False:
        return 10
    '\n    Returns list of retrieved top k videos based on the sims matrix\n    Args:\n        sims: similar matrix.\n        K: top k number of videos\n    '
    argm = np.argsort(-sims, axis=1)
    topk = argm[:, :k].reshape(-1)
    retrieved_videos = np.unique(topk)
    return retrieved_videos

def get_index_to_normalize(sims, videos):
    if False:
        return 10
    '\n    Returns list of indices to normalize from sims based on videos\n    Args:\n        sims: similar matrix.\n        videos: video array.\n    '
    argm = np.argsort(-sims, axis=1)[:, 0]
    result = np.array(list(map(lambda x: x in videos, argm)))
    result = np.nonzero(result)
    return result

def qb_norm(train_test, test_test, args):
    if False:
        while True:
            i = 10
    k = args.get('k', 1)
    beta = args.get('beta', 20)
    retrieved_videos = get_retrieved_videos(train_test, k)
    test_test_normalized = test_test
    train_test = np.exp(train_test * beta)
    test_test = np.exp(test_test * beta)
    normalizing_sum = np.sum(train_test, axis=0)
    index_for_normalizing = get_index_to_normalize(test_test, retrieved_videos)
    test_test_normalized[index_for_normalizing, :] = np.divide(test_test[index_for_normalizing, :], normalizing_sum)
    return test_test_normalized