import pandas as pd
import os

def load_file(filename):
    if False:
        for i in range(10):
            print('nop')
    '\n    Function for loading vectorised Twitter data (in data directory) into Pandas DataFrame\n    :param filename: str. path to text file in json format\n    :return: Pandas DataFrame\n    '
    data = pd.read_json(filename, orient='index')
    data['date'] = pd.to_datetime(data['date'], unit='s')
    return data

def get_files(data_directory='./data'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Function for getting list of text files\n    :param data_directory: path to directory of data\n    :return: list of directories\n    '
    dir_list = []
    if not data_directory.endswith('/'):
        data_directory += '/'
    for file in os.listdir(data_directory):
        if file.endswith('.txt'):
            dir_list.append(data_directory + file)
    return dir_list