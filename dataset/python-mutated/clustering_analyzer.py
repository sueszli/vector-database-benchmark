"""
This module implements methodologies to analyze clusters and determine whether they are poisonous.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Any, Dict, List, Tuple
import numpy as np
logger = logging.getLogger(__name__)

class ClusteringAnalyzer:
    """
    Class for all methodologies implemented to analyze clusters and determine whether they are poisonous.
    """

    @staticmethod
    def assign_class(clusters: np.ndarray, clean_clusters: np.ndarray, poison_clusters: np.ndarray) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines whether each data point in the class is in a clean or poisonous cluster\n\n        :param clusters: `clusters[i]` indicates which cluster the i'th data point is in.\n        :param clean_clusters: List containing the clusters designated as clean.\n        :param poison_clusters: List containing the clusters designated as poisonous.\n        :return: assigned_clean: `assigned_clean[i]` is a boolean indicating whether the ith data point is clean.\n        "
        assigned_clean = np.empty(np.shape(clusters))
        assigned_clean[np.isin(clusters, clean_clusters)] = 1
        assigned_clean[np.isin(clusters, poison_clusters)] = 0
        return assigned_clean

    def analyze_by_size(self, separated_clusters: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Designates as poisonous the cluster with less number of items on it.\n\n        :param separated_clusters: list where separated_clusters[i] is the cluster assignments for the ith class.\n        :return: all_assigned_clean, summary_poison_clusters, report:\n                 where all_assigned_clean[i] is a 1D boolean array indicating whether\n                 a given data point was determined to be clean (as opposed to poisonous) and\n                 summary_poison_clusters: array, where summary_poison_clusters[i][j]=1 if cluster j of class i was\n                 classified as poison, otherwise 0\n                 report: Dictionary with summary of the analysis\n        '
        report: Dict[str, Any] = {'cluster_analysis': 'smaller', 'suspicious_clusters': 0}
        all_assigned_clean = []
        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters: np.ndarray = np.zeros((nb_classes, nb_clusters), dtype=object)
        for (i, clusters) in enumerate(separated_clusters):
            sizes = np.bincount(clusters)
            total_dp_in_class = np.sum(sizes)
            poison_clusters = np.array([int(np.argmin(sizes))])
            clean_clusters = np.array(list(set(clusters) - set(poison_clusters)))
            for p_id in poison_clusters:
                summary_poison_clusters[i][p_id] = 1
            for c_id in clean_clusters:
                summary_poison_clusters[i][c_id] = 0
            assigned_clean = self.assign_class(clusters, clean_clusters, poison_clusters)
            all_assigned_clean.append(assigned_clean)
            report_class = {}
            for cluster_id in range(nb_clusters):
                ptc = sizes[cluster_id] / total_dp_in_class
                susp = cluster_id in poison_clusters
                dict_i = dict(ptc_data_in_cluster=round(ptc, 2), suspicious_cluster=susp)
                dict_cluster: Dict[str, Dict[str, int]] = {'cluster_' + str(cluster_id): dict_i}
                report_class.update(dict_cluster)
            report['Class_' + str(i)] = report_class
        report['suspicious_clusters'] = report['suspicious_clusters'] + np.sum(summary_poison_clusters)
        return (np.asarray(all_assigned_clean, dtype=object), summary_poison_clusters, report)

    def analyze_by_distance(self, separated_clusters: List[np.ndarray], separated_activations: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Assigns a cluster as poisonous if its median activation is closer to the median activation for another class\n        than it is to the median activation of its own class. Currently, this function assumes there are only two\n        clusters per class.\n\n        :param separated_clusters: list where separated_clusters[i] is the cluster assignments for the ith class.\n        :param separated_activations: list where separated_activations[i] is a 1D array of [0,1] for [poison,clean].\n        :return: all_assigned_clean, summary_poison_clusters, report:\n                 where all_assigned_clean[i] is a 1D boolean array indicating whether a given data point was determined\n                 to be clean (as opposed to poisonous) and summary_poison_clusters: array, where\n                 summary_poison_clusters[i][j]=1 if cluster j of class i was classified as poison, otherwise 0\n                 report: Dictionary with summary of the analysis.\n        '
        report: Dict[str, Any] = {'cluster_analysis': 0.0}
        all_assigned_clean = []
        cluster_centers = []
        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters = np.zeros((nb_classes, nb_clusters))
        for (_, activations) in enumerate(separated_activations):
            cluster_centers.append(np.median(activations, axis=0))
        for (i, (clusters, activation)) in enumerate(zip(separated_clusters, separated_activations)):
            clusters = np.array(clusters)
            cluster0_center = np.median(activation[np.where(clusters == 0)], axis=0)
            cluster1_center = np.median(activation[np.where(clusters == 1)], axis=0)
            cluster0_distance = np.linalg.norm(cluster0_center - cluster_centers[i])
            cluster1_distance = np.linalg.norm(cluster1_center - cluster_centers[i])
            cluster0_is_poison = False
            cluster1_is_poison = False
            dict_k = {}
            dict_cluster_0 = dict(cluster0_distance_to_its_class=str(cluster0_distance))
            dict_cluster_1 = dict(cluster1_distance_to_its_class=str(cluster1_distance))
            for (k, center) in enumerate(cluster_centers):
                if k == i:
                    pass
                else:
                    cluster0_distance_to_k = np.linalg.norm(cluster0_center - center)
                    cluster1_distance_to_k = np.linalg.norm(cluster1_center - center)
                    if cluster0_distance_to_k < cluster0_distance and cluster1_distance_to_k > cluster1_distance:
                        cluster0_is_poison = True
                    if cluster1_distance_to_k < cluster1_distance and cluster0_distance_to_k > cluster0_distance:
                        cluster1_is_poison = True
                    dict_cluster_0['distance_to_class_' + str(k)] = str(cluster0_distance_to_k)
                    dict_cluster_0['suspicious'] = str(cluster0_is_poison)
                    dict_cluster_1['distance_to_class_' + str(k)] = str(cluster1_distance_to_k)
                    dict_cluster_1['suspicious'] = str(cluster1_is_poison)
                    dict_k.update(dict_cluster_0)
                    dict_k.update(dict_cluster_1)
            report_class = dict(cluster_0=dict_cluster_0, cluster_1=dict_cluster_1)
            report['Class_' + str(i)] = report_class
            poison_clusters = []
            if cluster0_is_poison:
                poison_clusters.append(0)
                summary_poison_clusters[i][0] = 1
            else:
                summary_poison_clusters[i][0] = 0
            if cluster1_is_poison:
                poison_clusters.append(1)
                summary_poison_clusters[i][1] = 1
            else:
                summary_poison_clusters[i][1] = 0
            clean_clusters = np.array(list(set(clusters) - set(poison_clusters)))
            assigned_clean = self.assign_class(clusters, clean_clusters, np.array(poison_clusters))
            all_assigned_clean.append(assigned_clean)
        all_assigned_clean_array = np.asarray(all_assigned_clean, dtype=object)
        return (all_assigned_clean_array, summary_poison_clusters, report)

    def analyze_by_relative_size(self, separated_clusters: List[np.ndarray], size_threshold: float=0.35, r_size: int=2) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        if False:
            print('Hello World!')
        '\n        Assigns a cluster as poisonous if the smaller one contains less than threshold of the data.\n        This method assumes only 2 clusters\n\n        :param separated_clusters: List where `separated_clusters[i]` is the cluster assignments for the ith class.\n        :param size_threshold: Threshold used to define when a cluster is substantially smaller.\n        :param r_size: Round number used for size rate comparisons.\n        :return: all_assigned_clean, summary_poison_clusters, report:\n                 where all_assigned_clean[i] is a 1D boolean array indicating whether a given data point was determined\n                 to be clean (as opposed to poisonous) and summary_poison_clusters: array, where\n                 summary_poison_clusters[i][j]=1 if cluster j of class i was classified as poison, otherwise 0\n                 report: Dictionary with summary of the analysis.\n        '
        size_threshold = round(size_threshold, r_size)
        report: Dict[str, Any] = {'cluster_analysis': 'relative_size', 'suspicious_clusters': 0, 'size_threshold': size_threshold}
        all_assigned_clean = []
        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters = np.zeros((nb_classes, nb_clusters))
        for (i, clusters) in enumerate(separated_clusters):
            sizes = np.bincount(clusters)
            total_dp_in_class = np.sum(sizes)
            if np.size(sizes) > 2:
                raise ValueError(' RelativeSizeAnalyzer does not support more than two clusters.')
            percentages = np.round(sizes / float(np.sum(sizes)), r_size)
            poison_clusters = np.where(percentages < size_threshold)
            clean_clusters = np.where(percentages >= size_threshold)
            for p_id in poison_clusters[0]:
                summary_poison_clusters[i][p_id] = 1
            for c_id in clean_clusters[0]:
                summary_poison_clusters[i][c_id] = 0
            assigned_clean = self.assign_class(clusters, clean_clusters, poison_clusters)
            all_assigned_clean.append(assigned_clean)
            report_class = {}
            for cluster_id in range(nb_clusters):
                ptc = sizes[cluster_id] / total_dp_in_class
                susp = cluster_id in poison_clusters
                dict_i = dict(ptc_data_in_cluster=round(ptc, 2), suspicious_cluster=susp)
                dict_cluster = {'cluster_' + str(cluster_id): dict_i}
                report_class.update(dict_cluster)
            report['Class_' + str(i)] = report_class
        report['suspicious_clusters'] = report['suspicious_clusters'] + np.sum(summary_poison_clusters).item()
        return (np.asarray(all_assigned_clean), summary_poison_clusters, report)

    def analyze_by_silhouette_score(self, separated_clusters: list, reduced_activations_by_class: list, size_threshold: float=0.35, silhouette_threshold: float=0.1, r_size: int=2, r_silhouette: int=4) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        if False:
            print('Hello World!')
        "\n        Analyzes clusters to determine level of suspiciousness of poison based on the cluster's relative size\n        and silhouette score.\n        Computes a silhouette score for each class to determine how cohesive resulting clusters are.\n        A low silhouette score indicates that the clustering does not fit the data well, and the class can be considered\n        to be un-poisoned. Conversely, a high silhouette score indicates that the clusters reflect true splits in the\n        data.\n        The method concludes that a cluster is poison based on the silhouette score and the cluster relative size.\n        If the relative size is too small, below a size_threshold and at the same time\n        the silhouette score is higher than silhouette_threshold, the cluster is classified as poisonous.\n        If the above thresholds are not provided, the default ones will be used.\n\n        :param separated_clusters: list where `separated_clusters[i]` is the cluster assignments for the ith class.\n        :param reduced_activations_by_class: list where separated_activations[i] is a 1D array of [0,1] for\n               [poison,clean].\n        :param size_threshold: (optional) threshold used to define when a cluster is substantially smaller. A default\n        value is used if the parameter is not provided.\n        :param silhouette_threshold: (optional) threshold used to define when a cluster is cohesive. Default\n        value is used if the parameter is not provided.\n        :param r_size: Round number used for size rate comparisons.\n        :param r_silhouette: Round number used for silhouette rate comparisons.\n        :return: all_assigned_clean, summary_poison_clusters, report:\n                 where all_assigned_clean[i] is a 1D boolean array indicating whether a given data point was determined\n                 to be clean (as opposed to poisonous) summary_poison_clusters: array, where\n                 summary_poison_clusters[i][j]=1 if cluster j of class j was classified as poison\n                 report: Dictionary with summary of the analysis.\n        "
        from sklearn.metrics import silhouette_score
        size_threshold = round(size_threshold, r_size)
        silhouette_threshold = round(silhouette_threshold, r_silhouette)
        report: Dict[str, Any] = {'cluster_analysis': 'silhouette_score', 'size_threshold': str(size_threshold), 'silhouette_threshold': str(silhouette_threshold)}
        all_assigned_clean = []
        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters = np.zeros((nb_classes, nb_clusters))
        for (i, (clusters, activations)) in enumerate(zip(separated_clusters, reduced_activations_by_class)):
            bins = np.bincount(clusters)
            if np.size(bins) > 2:
                raise ValueError('Analyzer does not support more than two clusters.')
            percentages = np.round(bins / float(np.sum(bins)), r_size)
            poison_clusters = np.where(percentages < size_threshold)
            clean_clusters = np.where(percentages >= size_threshold)
            silhouette_avg = round(silhouette_score(activations, clusters), r_silhouette)
            dict_i: Dict[str, Any] = dict(sizes_clusters=str(bins), ptc_cluster=str(percentages), avg_silhouette_score=str(silhouette_avg))
            if np.shape(poison_clusters)[1] != 0:
                if silhouette_avg > silhouette_threshold:
                    clean_clusters = np.where(percentages < size_threshold)
                    logger.info('computed silhouette score: %s', silhouette_avg)
                    dict_i.update(suspicious=True)
                else:
                    poison_clusters = [[]]
                    clean_clusters = np.where(percentages >= 0)
                    dict_i.update(suspicious=False)
            else:
                dict_i.update(suspicious=False)
            report_class: Dict[str, Dict[str, bool]] = {'class_' + str(i): dict_i}
            for p_id in poison_clusters[0]:
                summary_poison_clusters[i][p_id] = 1
            for c_id in clean_clusters[0]:
                summary_poison_clusters[i][c_id] = 0
            assigned_clean = self.assign_class(clusters, clean_clusters, poison_clusters)
            all_assigned_clean.append(assigned_clean)
            report.update(report_class)
        return (np.asarray(all_assigned_clean), summary_poison_clusters, report)