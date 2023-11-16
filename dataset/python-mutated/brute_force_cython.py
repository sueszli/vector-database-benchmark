from typing import Callable, Dict
import brute_force_cython_ext

class BruteForceCython:
    """
    Class to perform search using a Brute force.
    """

    def __init__(self, hash_dict: Dict, distance_function: Callable) -> None:
        if False:
            print('Hello World!')
        '\n        Initialize a dictionary for mapping file names and corresponding hashes and a distance function to be used for\n        getting distance between two hash strings.\n\n        Args:\n            hash_dict: Dictionary mapping file names to corresponding hash strings {filename: hash}\n            distance_function:  A function for calculating distance between the hashes.\n        '
        self.distance_function = distance_function
        self.hash_dict = hash_dict

    def search(self, query: str, tol: int=10) -> Dict[str, int]:
        if False:
            i = 10
            return i + 15
        '\n        Function for searching using brute force.\n\n        Args:\n            query: hash string for which brute force needs to work.\n            tol: distance upto which duplicate is valid.\n\n        Returns:\n            List of tuples of the form [(valid_retrieval_filename1: distance), (valid_retrieval_filename2: distance)]\n        '
        filenames = []
        hash_vals = []
        for (filename, hash_val) in self.hash_dict.items():
            filenames.append(filename.encode('utf-8'))
            hash_vals.append(int(hash_val, 16))
        return brute_force_cython_ext.query(filenames, hash_vals, int(query, 16), tol)