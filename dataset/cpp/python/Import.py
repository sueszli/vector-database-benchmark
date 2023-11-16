import pandas as pd
import os


def load_file(filename):
    """
    Function for loading vectorised Twitter data (in data directory) into Pandas DataFrame
    :param filename: str. path to text file in json format
    :return: Pandas DataFrame
    """
    data = pd.read_json(filename, orient='index')

    data['date'] = pd.to_datetime(data['date'], unit='s')

    return data


def get_files(data_directory='./data'):
    """
    Function for getting list of text files
    :param data_directory: path to directory of data
    :return: list of directories
    """
    dir_list = []

    if not data_directory.endswith('/'):
        data_directory += '/'

    for file in os.listdir(data_directory):
        if file.endswith(".txt"):
            dir_list.append(data_directory+file)

    return dir_list
