"""
Scanning operations
"""
from typing import Callable, Tuple
import numpy as np

class ScanningOps:
    """
    Specific operations done during scanning
    """

    @staticmethod
    def optimize_in_single_dimension(pvalues: np.ndarray, a_max: float, image_to_node: bool, score_function: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) -> Tuple[float, np.ndarray, float]:
        if False:
            return 10
        '\n        Optimizes over all subsets of nodes for a given subset of images or over all subsets of images for a given\n        subset of nodes.\n\n        :param pvalues: pvalue ranges.\n        :param a_max: Determines the significance level threshold.\n        :param image_to_node: Informs the direction for optimization.\n        :param score_function: Scoring function.\n        :return: (best_score_so_far, subset, best_alpha).\n        '
        alpha_thresholds = np.unique(pvalues[:, :, 1])
        last_alpha_index = int(np.searchsorted(alpha_thresholds, a_max))
        alpha_thresholds = alpha_thresholds[0:last_alpha_index]
        step_for_50 = len(alpha_thresholds) / 50
        alpha_thresholds = alpha_thresholds[0::int(step_for_50) + 1]
        alpha_thresholds = np.append(alpha_thresholds, a_max)
        if image_to_node:
            number_of_elements = pvalues.shape[1]
            size_of_given = pvalues.shape[0]
            unsort_priority = np.zeros((pvalues.shape[1], alpha_thresholds.shape[0]))
        else:
            number_of_elements = pvalues.shape[0]
            size_of_given = pvalues.shape[1]
            unsort_priority = np.zeros((pvalues.shape[0], alpha_thresholds.shape[0]))
        for elem_indx in range(0, number_of_elements):
            if image_to_node:
                arg_sort_max = np.argsort(pvalues[:, elem_indx, 1])
                completely_included = np.searchsorted(pvalues[:, elem_indx, 1][arg_sort_max], alpha_thresholds, side='right')
            else:
                arg_sort_max = np.argsort(pvalues[elem_indx, :, 1])
                completely_included = np.searchsorted(pvalues[elem_indx, :, 1][arg_sort_max], alpha_thresholds, side='right')
            unsort_priority[elem_indx, :] = completely_included
        arg_sort_priority = np.argsort(-unsort_priority, axis=0)
        best_score_so_far = -10000
        best_alpha = -2
        alpha_count = 0
        for alpha_threshold in alpha_thresholds:
            alpha_v = np.ones(number_of_elements) * alpha_threshold
            n_alpha_v = np.cumsum(unsort_priority[:, alpha_count][arg_sort_priority][:, alpha_count])
            count_increments_this = np.ones(number_of_elements) * size_of_given
            n_v = np.cumsum(count_increments_this)
            vector_of_scores = score_function(n_alpha_v, n_v, alpha_v)
            best_score_for_this_alpha_idx = np.argmax(vector_of_scores)
            best_score_for_this_alpha = vector_of_scores[best_score_for_this_alpha_idx]
            if best_score_for_this_alpha > best_score_so_far:
                best_score_so_far = best_score_for_this_alpha
                best_size = best_score_for_this_alpha_idx + 1
                best_alpha = alpha_threshold
                best_alpha_count = alpha_count
            alpha_count = alpha_count + 1
        unsort = arg_sort_priority[:, best_alpha_count]
        subset = np.zeros(best_size).astype(int)
        for loc in range(0, best_size):
            subset[loc] = unsort[loc]
        return (best_score_so_far, subset, best_alpha)

    @staticmethod
    def single_restart(pvalues: np.ndarray, a_max: float, indices_of_seeds: np.ndarray, image_to_node: bool, score_function: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) -> Tuple[float, np.ndarray, np.ndarray, float]:
        if False:
            while True:
                i = 10
        '\n        Here we control the iteration between images->nodes and nodes->images. It starts with a fixed subset of nodes by\n        default.\n\n        :param pvalues: pvalue ranges.\n        :param a_max: Determines the significance level threshold.\n        :param indices_of_seeds: Indices of initial sets of images or nodes to perform optimization.\n        :param image_to_node: Informs the direction for optimization.\n        :param score_function: Scoring function.\n        :return: (best_score_so_far, best_sub_of_images, best_sub_of_nodes, best_alpha).\n        '
        best_score_so_far = -100000.0
        count = 0
        while True:
            if count == 0:
                if image_to_node:
                    sub_of_images = indices_of_seeds
                else:
                    sub_of_nodes = indices_of_seeds
            if image_to_node:
                (score_from_optimization, sub_of_nodes, optimal_alpha) = ScanningOps.optimize_in_single_dimension(pvalues[sub_of_images, :, :], a_max, image_to_node, score_function)
            else:
                (score_from_optimization, sub_of_images, optimal_alpha) = ScanningOps.optimize_in_single_dimension(pvalues[:, sub_of_nodes, :], a_max, image_to_node, score_function)
            if score_from_optimization > best_score_so_far:
                best_score_so_far = score_from_optimization
                best_sub_of_nodes = sub_of_nodes
                best_sub_of_images = sub_of_images
                best_alpha = optimal_alpha
                image_to_node = not image_to_node
                count = count + 1
            else:
                return (best_score_so_far, best_sub_of_images, best_sub_of_nodes, best_alpha)