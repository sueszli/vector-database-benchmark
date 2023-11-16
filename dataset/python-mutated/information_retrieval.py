from typing import List, Dict
import numpy as np

def avg_prec(correct_duplicates: List, retrieved_duplicates: List) -> float:
    if False:
        return 10
    '\n    Get average precision(AP) for a single query given correct and retrieved file names.\n\n    Args:\n        correct_duplicates: List of correct duplicates i.e., ground truth)\n        retrieved_duplicates: List of retrieved duplicates for one single query\n\n    Returns:\n        Average precision for this query.\n    '
    if len(retrieved_duplicates) == 0 and len(correct_duplicates) == 0:
        return 1.0
    if not len(retrieved_duplicates) or not len(correct_duplicates):
        return 0.0
    count_real_correct = len(correct_duplicates)
    relevance = np.array([1 if i in correct_duplicates else 0 for i in retrieved_duplicates])
    relevance_cumsum = np.cumsum(relevance)
    prec_k = [relevance_cumsum[k] / (k + 1) for k in range(len(relevance))]
    prec_and_relevance = [relevance[k] * prec_k[k] for k in range(len(relevance))]
    avg_precision = np.sum(prec_and_relevance) / count_real_correct
    return avg_precision

def ndcg(correct_duplicates: List, retrieved_duplicates: List) -> float:
    if False:
        print('Hello World!')
    '\n    Get Normalized discounted cumulative gain(NDCG) for a single query given correct and retrieved file names.\n\n    Args:\n        correct_duplicates: List of correct duplicates i.e., ground truth)\n        retrieved_duplicates: List of retrieved duplicates for one single query\n\n    Returns:\n        NDCG for this query.\n    '
    if len(retrieved_duplicates) == 0 and len(correct_duplicates) == 0:
        return 1.0
    if not len(retrieved_duplicates) or not len(correct_duplicates):
        return 0.0

    def dcg(rel):
        if False:
            print('Hello World!')
        relevance_numerator = [2 ** k - 1 for k in rel]
        relevance_denominator = [np.log2(k + 2) for k in range(len(rel))]
        dcg_terms = [relevance_numerator[k] / relevance_denominator[k] for k in range(len(rel))]
        dcg_at_k = np.sum(dcg_terms)
        return dcg_at_k
    relevance = np.array([1 if i in correct_duplicates else 0 for i in retrieved_duplicates])
    dcg_k = dcg(relevance)
    if dcg_k == 0:
        return 0.0
    idcg_k = dcg(sorted(relevance, reverse=True))
    return dcg_k / idcg_k

def jaccard_similarity(correct_duplicates: List, retrieved_duplicates: List) -> float:
    if False:
        print('Hello World!')
    '\n    Get jaccard similarity for a single query given correct and retrieved file names.\n\n    Args:\n        correct_duplicates: List of correct duplicates i.e., ground truth)\n        retrieved_duplicates: List of retrieved duplicates for one single query\n\n    Returns:\n        Jaccard similarity for this query.\n    '
    if len(retrieved_duplicates) == 0 and len(correct_duplicates) == 0:
        return 1.0
    if not len(retrieved_duplicates) or not len(correct_duplicates):
        return 0.0
    set_correct_duplicates = set(correct_duplicates)
    set_retrieved_duplicates = set(retrieved_duplicates)
    intersection_dups = set_retrieved_duplicates.intersection(set_correct_duplicates)
    union_dups = set_retrieved_duplicates.union(set_correct_duplicates)
    jacc_sim = len(intersection_dups) / len(union_dups)
    return jacc_sim

def mean_metric(ground_truth: Dict, retrieved: Dict, metric: str=None) -> float:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get mean of specified metric.\n\n    Args:\n        metric_func: metric function on which mean is to be calculated across all queries\n\n    Returns:\n        float representing mean of the metric across all queries\n    '
    metric = metric.lower()
    metric_lookup = {'map': avg_prec, 'ndcg': ndcg, 'jaccard': jaccard_similarity}
    metric_func = metric_lookup[metric]
    metric_vals = []
    for k in ground_truth.keys():
        metric_vals.append(metric_func(ground_truth[k], retrieved[k]))
    return np.mean(metric_vals)

def get_all_metrics(ground_truth: Dict, retrieved: Dict) -> Dict:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get mean of all information retrieval metrics across all queries.\n\n    Args:\n        ground_truth: A dictionary representing ground truth with filenames as key and a list of duplicate filenames\n        as value.\n        retrieved: A dictionary representing retrieved duplicates with filenames as key and a list of retrieved\n        duplicate filenames as value.\n\n    Returns:\n        Dictionary of all mean metrics.\n    '
    all_average_metrics = {'map': mean_metric(ground_truth, retrieved, metric='map'), 'ndcg': mean_metric(ground_truth, retrieved, metric='ndcg'), 'jaccard': mean_metric(ground_truth, retrieved, metric='jaccard')}
    return all_average_metrics