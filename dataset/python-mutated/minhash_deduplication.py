import json
import multiprocessing as mp
import re
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Set, Tuple, Type
from datasets import Dataset
from datasketch import MinHash, MinHashLSH
from dpu_utils.utils.iterators import ThreadedIterator
from tqdm import tqdm
NON_ALPHA = re.compile('[^A-Za-z_0-9]')
MIN_NUM_TOKENS = 10
NUM_PERM = 256

def get_min_hash(tokens: List[str]) -> Optional[MinHash]:
    if False:
        i = 10
        return i + 15
    'Compute the MinHash of a code snippet.'
    if len(tokens) < MIN_NUM_TOKENS:
        return None
    min_hash = MinHash(num_perm=NUM_PERM)
    for token in set(tokens):
        min_hash.update(token.encode())
    return min_hash

def get_tokens(code: str) -> Set[str]:
    if False:
        while True:
            i = 10
    'Tokenize a code snippet.'
    return {t for t in NON_ALPHA.split(code) if len(t.strip()) > 0}

class DuplicationIndex:

    def __init__(self, *, duplication_jaccard_threshold: float=0.85):
        if False:
            print('Hello World!')
        self._duplication_jaccard_threshold = duplication_jaccard_threshold
        self._num_perm = NUM_PERM
        self._index = MinHashLSH(threshold=self._duplication_jaccard_threshold, num_perm=self._num_perm)
        self._duplicate_clusters = defaultdict(set)

    def add(self, code_key: Tuple, min_hash: MinHash) -> None:
        if False:
            i = 10
            return i + 15
        'Add a key to _index (MinHashLSH)\n        the min_hash is used to query closest matches based on the jaccard_threshold.\n        The new key is either added to a existing cluster of one close match,\n        or a new cluster is created. The clusters created in this way, depend on the order of add.\n\n        Args:\n            code_key (Tuple of (index, repo_name, path)):\n                Theoritically any hasbale key. Here we use a tuple to retrieve the information later.\n            min_hash: MinHash of the code_key.\n        '
        close_duplicates = self._index.query(min_hash)
        if code_key in self._index.keys:
            print(f'Duplicate key {code_key}')
            return
        self._index.insert(code_key, min_hash)
        if len(close_duplicates) > 0:
            for base_duplicate in close_duplicates:
                if base_duplicate in self._duplicate_clusters:
                    self._duplicate_clusters[base_duplicate].add(code_key)
                    break
            else:
                self._duplicate_clusters[close_duplicates[0]].add(code_key)

    def get_duplicate_clusters(self) -> List[List[Dict]]:
        if False:
            i = 10
            return i + 15
        'Export the duplicate clusters.\n        For each cluster, the first element is the base element of the cluster.\n        The base element has an estimation jaccard similarity higher than the threshold with all the other elements.\n\n        Returns:\n            duplicate_clusters (List[List[Dict]]):\n                List of duplicate clusters.\n        '
        duplicate_clusters = []
        for (base, duplicates) in self._duplicate_clusters.items():
            cluster = [base] + list(duplicates)
            cluster = [{'base_index': el[0], 'repo_name': el[1], 'path': el[2]} for el in cluster]
            duplicate_clusters.append(cluster)
        return duplicate_clusters

    def save(self, filepath) -> None:
        if False:
            i = 10
            return i + 15
        duplicate_clusters = self.get_duplicate_clusters()
        with open(filepath, 'w') as f:
            json.dump(duplicate_clusters, f)

def _compute_min_hash(element):
    if False:
        while True:
            i = 10
    (index, data) = element
    min_hash = get_min_hash([t for t in NON_ALPHA.split(data['content']) if len(t.strip()) > 0])
    if min_hash is not None:
        return ((index, data['repo_name'], data['path']), min_hash)

def minhash_iter(dataset_iterator: Type[Dataset]):
    if False:
        i = 10
        return i + 15
    with mp.Pool() as pool:
        for data in pool.imap_unordered(_compute_min_hash, ThreadedIterator(dataset_iterator, max_queue_size=10000), chunksize=100):
            if data is not None:
                yield data

def make_duplicate_clusters(dataset_iterator: Type[Dataset], jaccard_threshold: float):
    if False:
        print('Hello World!')
    'Find duplicate clusters in the dataset in two steps:\n    1. Compute MinHash for each code snippet. MinHash is a tool for fast jaccard similarity estimation.\n    This step is computed using an asynchronous multiprocessing pool, minhash_iter\n    2. Find duplicate clusters. The computed MinHash is added sequentially to the DuplicationIndex.\n    This step cannot be parallelized. So using asynchronous thread in the previous step helps to speed up the process.\n    '
    di = DuplicationIndex(duplication_jaccard_threshold=jaccard_threshold)
    for (filename, min_hash) in tqdm(ThreadedIterator(minhash_iter(enumerate(dataset_iterator)), max_queue_size=100)):
        di.add(filename, min_hash)
    return di.get_duplicate_clusters()

def jaccard_similarity(code1: str, code2: str) -> float:
    if False:
        while True:
            i = 10
    'Compute the Jaccard similarity of two code snippets.'
    tokens1 = get_tokens(code1)
    tokens2 = get_tokens(code2)
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)
_shared_dataset = None

def _find_cluster_extremes_shared(cluster, jaccard_threshold):
    if False:
        for i in range(10):
            print('nop')
    'Find a reduced cluster such that each code in the origin cluster is similar to at least one code in the reduced cluster.\n    Two codes are similar if their Jaccard similarity is above the threshold.\n\n    Args:\n        cluster (List[dict]):\n           cluster is a list of dict, each dict contains the following keys:\n                - base_index\n                - repo_name\n                - path\n            This is a typical output of DuplicationIndex.get_duplicate_clusters()\n        jaccard_threshold (float):\n            threshold for Jaccard similarity.\n            Two codes are similar if their Jaccard similarity is above the threshold.\n\n    Returns:\n        extremes (List[dict]):\n            A reduced representation of the cluster. The field copies is added to each dict.\n            The copies field indicates the number of similar codes in the cluster for a extreme.\n    '
    extremes = []
    for element1 in cluster:
        code1 = _shared_dataset[element1['base_index']]['content']
        for element2 in extremes:
            code2 = _shared_dataset[element2['base_index']]['content']
            if jaccard_similarity(code1, code2) >= jaccard_threshold:
                element2['copies'] += 1
                break
        else:
            element1['copies'] = 1
            extremes.append(element1)
    return extremes

def find_extremes(cluster_list, dataset, jaccard_threshold):
    if False:
        while True:
            i = 10
    'Call the _find_cluster_extremes_shared function in a parallel fashion.\n\n    Args:\n        cluster_list (List[List[Dict]]):\n            each cluster is a list of dicts with the key base_index,\n            referring to the index of the base code in the dataset.\n        dataset (Type[Dataset]):\n            dataset is used to access the content of the code snippets,\n            using the base_index from the cluster_list.\n            dataset is shared between all the processes using a glabal variable (any other way to share the dataset?),\n            otherwise the multi processing is not speeded up.\n        jaccard_threshold (float):\n            the threshold for the jaccard similarity. The default value is 0.85\n\n    Returns:\n        extremes_list (List[Dict]):\n            Each cluster is reduced to extremes.\n            See _find_cluster_extremes_shared for the definition of extremes.\n    '
    global _shared_dataset
    _shared_dataset = dataset
    extremes_list = []
    f = partial(_find_cluster_extremes_shared, jaccard_threshold=jaccard_threshold)
    with mp.Pool() as pool:
        for extremes in tqdm(pool.imap_unordered(f, cluster_list), total=len(cluster_list)):
            extremes_list.append(extremes)
    return extremes_list

def deduplicate_dataset(dataset: Type[Dataset], jaccard_threshold: float=0.85) -> Tuple[Type[Dataset], List[List[Dict]]]:
    if False:
        print('Hello World!')
    'Deduplicate the dataset using minhash and jaccard similarity.\n    This function first generate duplicate clusters, then each cluster\n    is reduced to the extremes that are similar to the other elements in the cluster.\n    Codes are called similar if their Jaccard similarity is greater than jaccard_threshold (0.85 default).\n\n    Args:\n        dataset (Type[Dataset]):\n            The dataset to deduplicate.\n        jaccard_threshold (float, default=0.85):\n            jaccard threshold to determine if two codes are similar\n\n    Returns:\n        ds_dedup (Type[Dataset]):\n            The deduplicated dataset.\n        duplicate_clusters (List[List[Dict]]):\n            The list of duplicate clusters.\n            Each cluster is a list of dicts with the following keys:\n            - base_index : int\n                The index of the code in the original dataset.\n            - repo_name : str\n            - path : str\n            - copies : int\n                The number of copies of the code in the cluster. (find_cluster_extremes)\n            - is_extreme : bool\n                Whether the code is an extreme in the cluster.\n            All the codes in the cluster are removed from the dataset except the extremes.\n\n    Example:\n        >>> from datasets import load_dataset\n        >>> from minhash_deduplication import deduplicate_dataset\n        >>> ds = load_dataset("lvwerra/codeparrot-clean", split="train")\n        >>> ds_dedup, duplicate_clusters = deduplicate_dataset(ds, jaccard_threshold=0.85)\n    '
    duplicate_clusters = make_duplicate_clusters(dataset, jaccard_threshold)
    duplicate_indices = {x['base_index'] for cluster in duplicate_clusters for x in cluster}
    extreme_dict = {}
    extremes_clusters = find_extremes(duplicate_clusters, dataset, jaccard_threshold)
    for extremes in extremes_clusters:
        for element in extremes:
            extreme_dict[element['base_index']] = element
    remove_indices = duplicate_indices - set(extreme_dict.keys())
    ds_filter = dataset.filter(lambda x, idx: idx not in remove_indices, with_indices=True)
    for cluster in duplicate_clusters:
        for element in cluster:
            element['is_extreme'] = element['base_index'] in extreme_dict
            if element['is_extreme']:
                element['copies'] = extreme_dict[element['base_index']]['copies']
    print(f'Original dataset size: {len(dataset)}')
    print(f'Number of duplicate clusters: {len(duplicate_clusters)}')
    print(f'Files in duplicate cluster: {len(duplicate_indices)}')
    print(f'Unique files in duplicate cluster: {len(extreme_dict)}')
    print(f'Filtered dataset size: {len(ds_filter)}')
    return (ds_filter, duplicate_clusters)