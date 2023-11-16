import numpy as np
import pandas as pd
import requests

def load_pandas_df(azure_storage_account_name='azureopendatastorage', azure_storage_sas_token='', container_name='covid19temp', metadata_filename='metadata.csv'):
    if False:
        for i in range(10):
            print('nop')
    'Loads the Azure Open Research COVID-19 dataset as a pd.DataFrame.\n\n    The Azure COVID-19 Open Research Dataset may be found at https://azure.microsoft.com/en-us/services/open-datasets/catalog/covid-19-open-research/\n\n    Args:\n        azure_storage_account_name (str): Azure storage account name.\n        azure_storage_sas_token (str): Azure storage SAS token.\n        container_name (str): Azure storage container name.\n        metadata_filename (str): Name of file containing top-level metadata for the dataset.\n\n    Returns:\n        metadata (pandas.DataFrame): Metadata dataframe.\n    '
    uri = 'https://{acct}.blob.core.windows.net/{container}/{filename}{sas}'.format(acct=azure_storage_account_name, container=container_name, filename=metadata_filename, sas=azure_storage_sas_token)
    return pd.read_csv(uri)

def remove_duplicates(df, cols):
    if False:
        print('Hello World!')
    'Remove duplicated entries.\n\n    Args:\n        df (pd.DataFrame): Pandas dataframe.\n        cols (list of str): Name of columns in which to look for duplicates.\n\n    Returns:\n        df (pandas.DataFrame): Pandas dataframe with duplicate rows dropped.\n\n    '
    for col in cols:
        df = df.reset_index(drop=True)
        dup_rows = np.where(df.duplicated([col]))[0]
        df = df.drop(dup_rows)
    return df

def remove_nan(df, cols):
    if False:
        while True:
            i = 10
    'Remove rows with NaN values in specified column.\n\n    Args:\n        df (pandas.DataFrame): Pandas dataframe.\n        cols (list of str): Name of columns in which to look for NaN.\n\n    Returns:\n        df (pandas.DataFrame): Pandas dataframe with invalid rows dropped.\n\n    '
    for col in cols:
        df[col].replace('', np.nan, inplace=True)
        df = df[df[col].notna()]
    return df

def clean_dataframe(df):
    if False:
        print('Hello World!')
    'Clean up the dataframe.\n\n    Args:\n        df (pandas.DataFrame): Pandas dataframe.\n\n    Returns:\n        df (pandas.DataFrame): Cleaned pandas dataframe.\n    '
    cols = ['cord_uid', 'doi']
    df = remove_duplicates(df, cols)
    cols = ['cord_uid', 'doi', 'title', 'license', 'url']
    df = remove_nan(df, cols)
    return df

def retrieve_text(entry, container_name, azure_storage_account_name='azureopendatastorage', azure_storage_sas_token=''):
    if False:
        return 10
    'Retrieve body text from article of interest.\n\n    Args:\n        entry (pd.Series): A single row from the dataframe (df.iloc[n]).\n        container_name (str): Azure storage container name.\n        azure_storage_account_name (str): Azure storage account name.\n        azure_storage_sas_token (str): Azure storage SAS token.\n\n    Results:\n        text (str): Full text of the blob as a single string.\n    '
    try:
        filename = entry['pdf_json_files'] or entry['pmc_json_files']
        uri = 'https://{acct}.blob.core.windows.net/{container}/{filename}{sas}'.format(acct=azure_storage_account_name, container=container_name, filename=filename, sas=azure_storage_sas_token)
        data = requests.get(uri, headers={'Content-type': 'application/json'}).json()
        text = ' '.join([paragraph['text'] for paragraph in data['body_text']])
    except Exception:
        text = ''
    return text

def get_public_domain_text(df, container_name, azure_storage_account_name='azureopendatastorage', azure_storage_sas_token=''):
    if False:
        for i in range(10):
            print('nop')
    'Get all public domain text.\n\n    Args:\n        df (pandas.DataFrame): Metadata dataframe for public domain text.\n        container_name (str): Azure storage container name.\n        azure_storage_account_name (str): Azure storage account name.\n        azure_storage_sas_token (str): Azure storage SAS token.\n\n    Returns:\n        df_full (pandas.DataFrame): Dataframe with select metadata and full article text.\n    '
    df = df.reset_index(drop=True)
    df['full_text'] = df.apply(lambda row: retrieve_text(row, container_name, azure_storage_account_name, azure_storage_sas_token), axis=1)
    empty_rows = np.where(df['full_text'] == '')[0]
    df = df.drop(empty_rows)
    df_full = df[['cord_uid', 'doi', 'title', 'publish_time', 'authors', 'journal', 'url', 'abstract', 'full_text']]
    df_full = df_full.reset_index()
    return df_full