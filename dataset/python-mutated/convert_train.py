import os
from os import listdir
from argparse import ArgumentParser
import pandas as pd
from urllib.parse import urlparse
from os.path import exists
from bigdl.dllib.utils import log4Error

def is_local_and_existing_uri(uri):
    if False:
        while True:
            i = 10
    parsed_uri = urlparse(uri)
    log4Error.invalidInputError(not parsed_uri.scheme or parsed_uri.scheme == 'file', 'Not Local File!')
    log4Error.invalidInputError(not parsed_uri.netloc or parsed_uri.netloc.lower() == 'localhost', 'Not Local File!')
    log4Error.invalidInputError(exists(parsed_uri.path), 'File Not Exist!')

def _parse_args():
    if False:
        i = 10
        return i + 15
    parser = ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the folder of parquet files.')
    parser.add_argument('--output_folder', type=str, default='.', help='The path to save the preprocessed data to parquet files. ')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = _parse_args()
    input_files = [f for f in listdir(args.input_folder) if f.endswith('.parquet')]
    for f in input_files:
        is_local_and_existing_uri(os.path.join(args.input_folder, f))
        df = pd.read_parquet(os.path.join(args.input_folder, f))
        df = df.rename(columns={'text_ tokens': 'text_tokens'})
        df = df.rename(columns={'retweet_timestampe': 'retweet_timestamp'})
        df.to_parquet(os.path.join(args.output_folder, '%s' % f))