import copy
from typing import Callable, Dict, Tuple

class BkTreeNode:
    """
    Class to contain the attributes of a single node in the BKTree.
    """

    def __init__(self, node_name: str, node_value: str, parent_name: str=None) -> None:
        if False:
            while True:
                i = 10
        self.node_name = node_name
        self.node_value = node_value
        self.parent_name = parent_name
        self.children = {}

class BKTree:
    """
    Class to construct and perform search using a BKTree.
    """

    def __init__(self, hash_dict: Dict, distance_function: Callable) -> None:
        if False:
            while True:
                i = 10
        '\n        Initialize a root for the BKTree and triggers the tree construction using the dictionary for mapping file names\n        and corresponding hashes.\n\n        Args:\n            hash_dict: Dictionary mapping file names to corresponding hash strings {filename: hash}\n            distance_function: A function for calculating distance between the hashes.\n        '
        self.hash_dict = hash_dict
        self.distance_function = distance_function
        self.all_keys = list(self.hash_dict.keys())
        self.ROOT = self.all_keys[0]
        self.all_keys.remove(self.ROOT)
        self.dict_all = {self.ROOT: BkTreeNode(self.ROOT, self.hash_dict[self.ROOT])}
        self.candidates = [self.dict_all[self.ROOT].node_name]
        self.construct_tree()

    def _insert_in_tree(self, k: str, current_node: str) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Function to insert a new node into the BKTree.\n\n        Args:\n            k: filename for inserting into the BKTree.\n            current_node: Node of the tree to which the new node should be added.\n\n        Return:\n            0 for successful execution.\n        '
        dist_current_node = self.distance_function(self.hash_dict[k], self.dict_all[current_node].node_value)
        condition_insert_current_node_child = not self.dict_all[current_node].children or dist_current_node not in list(self.dict_all[current_node].children.values())
        if condition_insert_current_node_child:
            self.dict_all[current_node].children[k] = dist_current_node
            self.dict_all[k] = BkTreeNode(k, self.hash_dict[k], parent_name=current_node)
        else:
            for (i, val) in self.dict_all[current_node].children.items():
                if val == dist_current_node:
                    node_to_add_to = i
                    break
            self._insert_in_tree(k, node_to_add_to)
        return 0

    def construct_tree(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct the BKTree.\n        '
        for k in self.all_keys:
            self._insert_in_tree(k, self.ROOT)

    def _get_next_candidates(self, query: str, candidate_obj: BkTreeNode, tolerance: int) -> Tuple[list, int, float]:
        if False:
            return 10
        '\n        Get candidates for checking if the query falls within the distance tolerance. Sets a validity flag if the input\n        candidate BKTree node is valid (distance to this candidate is within the distance tolerance from the query.)\n\n        Args:\n            query: The hash for which retrievals are needed.\n            candidate_obj: A BKTree object which is a candidate for being checked as valid.\n            tolerance: Distance within which the candidate is considered valid.\n\n        Returns:\n            new candidates to examine, validity flag indicating whether current candidate is within the distance\n            tolerance, distance of the current candidate from the query hash.\n        '
        dist = self.distance_function(candidate_obj.node_value, query)
        if dist <= tolerance:
            validity = 1
        else:
            validity = 0
        search_range_dist = list(range(dist - tolerance, dist + tolerance + 1))
        candidate_children = candidate_obj.children
        candidates = [k for k in candidate_children.keys() if candidate_children[k] in search_range_dist]
        return (candidates, validity, dist)

    def search(self, query: str, tol: int=5) -> Dict:
        if False:
            return 10
        '\n        Function to search the bktree given a hash of the query image.\n\n        Args:\n            query: hash string for which BKTree needs to be searched.\n            tol: distance upto which duplicate is valid.\n\n        Returns:\n            List of tuples of the form [(valid_retrieval_filename1: distance), (valid_retrieval_filename2: distance)]\n        '
        valid_retrievals = []
        candidates_local = copy.deepcopy(self.candidates)
        while len(candidates_local) != 0:
            candidate_name = candidates_local.pop()
            (cand_list, valid_flag, dist) = self._get_next_candidates(query, self.dict_all[candidate_name], tolerance=tol)
            if valid_flag:
                valid_retrievals.append((candidate_name, int(dist)))
            candidates_local.extend(cand_list)
        return valid_retrievals