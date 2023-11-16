import bz2
import gzip
import json
import pickle
from pathlib import Path
import numpy as np
import polars as pl
from tqdm import tqdm
path_string = 'PUT THE PATH HERE TO WHERE YOU DOWNLOADED AND EXTRACTED THE ARCHIVE .TAR'
folder_path = Path(path_string)
file_list_pkl = folder_path / 'file_list.pkl'
processed_file_list_pkl = folder_path / 'processed_file_list.pkl'
processed_folder_path = folder_path / 'processed'
processed_folder_path.mkdir(parents=True, exist_ok=True)
processed_max_buffer = 5000
wanted_cols = ['timestamp_ms', 'id', 'text', 'truncated', 'in_reply_to_status_id', 'in_reply_to_user_id', 'is_quote_status', 'quote_count', 'reply_count', 'retweet_count', 'favorite_count', 'filter_level', 'lang', 'possibly_sensitive', 'hashtags', 'user_id', 'user_verified', 'user_followers_count', 'user_statuses_count']

def main(file_list_pkl, folder_path, processed_max_buffer):
    if False:
        for i in range(10):
            print('nop')
    '\n    Runs the main processing script to get files, loop through them, and process them.\n    Outputs larger json.gz files made by concat the pre-filtered dataframes from\n    the original json.gz files.\n    '
    file_list = get_file_paths(file_list_pkl, folder_path)
    process_json(file_list, processed_max_buffer)
    print('Done')

def get_file_paths(file_list_pkl, folder_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Gets the file paths by recursively checking the folder structure.\n    # Based on code from stackoverflow https://stackoverflow.com/questions/26835477/pickle-load-variable-if-exists-or-create-and-save-it\n    '
    try:
        allpaths = pickle.load(open(file_list_pkl, 'rb'))
    except (OSError, IOError) as e:
        print(e)
        allpaths = sorted(list(folder_path.rglob('*.[gz bz2]*')))
        pickle.dump(allpaths, open(file_list_pkl, 'wb'))
    print('Got file paths.')
    return allpaths

def get_processed_list(processed_file_list_pkl):
    if False:
        i = 10
        return i + 15
    try:
        processed_list = pickle.load(open(processed_file_list_pkl, 'rb'))
    except (OSError, IOError) as e:
        print(e)
        processed_list = []
        pickle.dump(processed_list, open(processed_file_list_pkl, 'wb'))
    return processed_list

def modify_dict_cols(j_dict):
    if False:
        print('Hello World!')
    j_dict['user_id'] = np.int64(j_dict['user']['id'])
    j_dict['user_followers_count'] = np.int64(j_dict['user']['followers_count'])
    j_dict['user_statuses_count'] = np.int64(j_dict['user']['statuses_count'])
    j_dict['hashtags'] = [h['text'] for h in j_dict['entities']['hashtags']]
    j_dict['id'] = np.int64(j_dict['id'])
    try:
        j_dict['in_reply_to_status_id'] = np.int64(j_dict['in_reply_to_status_id'])
    except Exception as e:
        print(e)
        j_dict['in_reply_to_status_id'] = j_dict['in_reply_to_status_id']
    try:
        j_dict['in_reply_to_user_id'] = np.int64(j_dict['in_reply_to_user_id'])
    except Exception as e:
        print(e)
        j_dict['in_reply_to_user_id'] = j_dict['in_reply_to_user_id']
    for key in wanted_cols:
        if key not in j_dict:
            j_dict[key] = None
    j_dict = {key: j_dict[key] for key in wanted_cols}
    return j_dict

def process_single_file(f, processed_list):
    if False:
        print('Hello World!')
    j_dict_list = []
    if f not in processed_list:
        if f.suffix == '.bz2':
            with bz2.BZ2File(f) as file:
                for line in file:
                    j_dict = json.loads(line)
                    if 'delete' not in j_dict:
                        if j_dict['truncated'] is False:
                            j_dict = modify_dict_cols(j_dict)
                            j_dict_list.append(j_dict)
        else:
            with gzip.open(f, 'r') as file:
                for line in file:
                    j_dict = json.loads(line)
                    if 'delete' not in j_dict:
                        if j_dict['truncated'] is False:
                            j_dict = modify_dict_cols(j_dict)
                            j_dict_list.append(j_dict)
        return j_dict_list

def process_json(file_list, processed_max_buffer):
    if False:
        return 10
    '\n    Loops through file list and loads the compressed\n    json into a list of dicts after some pre-processing.\n\n    Makes sure dicts are ordered in a specific\n    way to make sure polars can read them.\n    '
    processed_list = get_processed_list(processed_file_list_pkl)
    j_list = []
    temp_processed_files = []
    for (i, f) in enumerate(tqdm(file_list)):
        j_dict_list = process_single_file(f, processed_list)
        j_list.extend(j_dict_list)
        temp_processed_files.append(f)
        if len(temp_processed_files) == processed_max_buffer:
            processed_file_name = f'processed_json_{i}.parquet'
            processed_file_path = processed_folder_path / processed_file_name
            pl.DataFrame(j_list, columns=wanted_cols).write_parquet(processed_file_path)
            processed_list.extend(temp_processed_files)
            pickle.dump(processed_list, open(processed_file_list_pkl, 'wb'))
            j_list = []
            temp_processed_files = []
    processed_file_name = f'processed_json_{i}.parquet'
    processed_file_path = processed_folder_path / processed_file_name
    pl.from_dicts(j_dict_list).write_parquet(processed_file_path)
    processed_list.extend(temp_processed_files)
    pickle.dump(processed_list, open(processed_file_list_pkl, 'wb'))
    j_dict_list = []
    temp_processed_files = []
    print('Processing completed')
if __name__ == '__main__':
    main(file_list_pkl, folder_path, processed_max_buffer)