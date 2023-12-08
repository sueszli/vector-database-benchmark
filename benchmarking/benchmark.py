import json
from milvusdb.milvus_db import MilvusDB
import multiprocessing
import time
import os
from tqdm import tqdm

def read_combined_json(file_path):
    """
    Reads a combined JSON file into memory.

    :param file_path: The path to the combined JSON file.
    :return: The data loaded from the JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data

    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {file_path}.")
        raise

# Example usage:
# data = read_combined_json('path_to_combined_json_file.json')

def count_greater_than_min(first_list, second_list):
    """
    Count how many elements in the second list are greater than the smallest element in the first list.

    Args:
    first_list (list of float): The first list of floats.
    second_list (list of float): The second list of floats.

    Returns:
    int: The count of elements in the second list that are greater than the minimum of the first list.
    """
    if not first_list:  # Check if the first list is empty
        return 0

    min_first_list = min(first_list)  # Find the minimum in the first list
    count = 0  # Initialize the counter

    # Count elements in the second list that are greater than the minimum of the first list
    for element in second_list:
        if element > min_first_list:
            count += 1

    return count

def compute_recall_for_one(query_result, ground_truth):
    """
    Computes the recall for a single query result.

    :param query_result: The query result.
    :param ground_truth: The ground truth.
    :return: The recall.
    """
    #
    return count_greater_than_min(ground_truth, query_result) / len(ground_truth)


def compute_avg_recall(query_results, ground_truths):
    """
    Computes the average recall for a set of query results.

    :param query_results: The query results.
    :param ground_truths: The ground truths.
    :return: The average recall.
    """

    recalls = []
    for query_result, ground_truth in zip(query_results, ground_truths):
        recalls.append(compute_recall_for_one(query_result, ground_truth))
    if not recalls:
        return 0

    return sum(recalls) / len(recalls)


# def write_to_db(dbobj, collection_name, combined_json):
#     for file in combined_json:
#         for func in file:
#             # func has dict_keys(['func_name', 'original', 'mutated', 'original_embedding', 'mutated_embeddings'])
#             dbobj.insert_item(collection_name, func['mutated_embeddings'])

import numpy as np
def compute_cosine_similarity(vec1, vec2):
    return (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))).tolist()[0]



def process_file(dbobj, collection_name, list, noofthreads):
    print("thread started")
    start = time.time()
    t=0
    for file in list:
        for func in file:
            # print(len(func['mutated_embeddings'])) -> 5
            # print(len(func['mutated_embeddings'][0][0])) -> 1024
            for emb in func['mutated_embeddings']:
                t+=1
                dbobj.insert_item(collection_name, [emb])
    if t:
        print(f"thread {noofthreads} time taken/upload: {(time.time() - start)/t} seconds")

def write_to_db(dbobj, collection_name, combined_json, num_threads):

    # Create a multiprocessing pool with the specified number of threads
    with multiprocessing.Pool(num_threads) as pool:
        print(f"Number of threads: {num_threads}")
        # Map the process_file function to each file in the combined_json
        pool.starmap(process_file, [(dbobj, collection_name, file, num_threads) for file in combined_json])

def searcher(dbobj, collection_name, list, noofthreads):
    print("thread started")
    start = time.time()
    t=0
    results = []

    for file in list:
        og = []
        for func in file:
            # print(len(func['mutated_embeddings'])) -> 5
            # print(len(func['mutated_embeddings'][0][0])) -> 1024
            t+=1
            og.append(dbobj.query_item(collection_name, func['original_embedding'][0]))
        results.append(og)
    if t:
        print(f"thread {noofthreads} time taken/search: {(time.time() - start)/t} seconds")

    x = []
    y = []
    for k, file in enumerate(list):
        for kk, func in enumerate(file):
            cosinesim = []
            calc = []
            for emb in func['mutated_embeddings']:
                cosinesim.append(compute_cosine_similarity(emb, func['original_embedding'][0]))
            for hit in results[k][kk]:
                calc.append(hit.distance*100)
            x.append(cosinesim)
            y.append(calc)
            # print(cosinesim, calc)
    print(f"thread {noofthreads} recall: {compute_avg_recall(y, x)}")


def search_in_db(dbobj, collection_name, combined_json, num_threads):
    # Create a multiprocessing pool with the specified number of threads
    with multiprocessing.Pool(num_threads) as pool:
        print(f"Number of threads: {num_threads}")
        # Map the process_file function to each file in the combined_json
        pool.starmap(searcher, [(dbobj, collection_name, file, num_threads) for file in combined_json])

def main(dbobj, dir_path):
    threads_per_file = [16,10, 1, 2, 3, 4, 6, 8, 12, 20, 32, 64]
    for index, num_threads in enumerate(threads_per_file):
        file_path = os.path.join(dir_path, f'combined_mutated_with_embdngs_{index + 1}.json')

        combined_json = read_combined_json(file_path)
        indexname = f'bm_11_{index + 1}'
        list_of_data = []
        for i in range(num_threads):
            list_of_data.append(combined_json[i*100::(i+1)*100])
        write_to_db(dbobj, indexname, list_of_data, num_threads)
        dbobj.index(indexname)
        search_in_db(dbobj, indexname, list_of_data, num_threads)

def read_one_thread(dbobj, collection_name, index):
    file_path = os.path.join(dir_path, f'combined_mutated_with_embdngs_{index + 1}.json')
    combined_json = read_combined_json(file_path)
    combined_json = combined_json[:1000]
    n = 0
    start = time.time()
    dbobj.index(collection_name)
    for file in combined_json:
        for func in file:
            # print(func['original_embedding'])
            outp = dbobj.query_item(collection_name, func['original_embedding'][0])
            n += 1
            if n == 100:
                print(f"Time taken for 1000 queries: {time.time() - start} seconds")
                break



if __name__ == '__main__':
    dbobj = MilvusDB()
    dir_path = '/Users/vikram/Desktop/School/M2/CS854/vdb_benchmarking_dataset/'
    main(dbobj, dir_path)
    # read_one_thread(dbobj, 'bma_8_4', 1)
