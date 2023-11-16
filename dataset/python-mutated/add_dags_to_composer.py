from __future__ import annotations
import argparse
import glob
import os
from shutil import copytree, ignore_patterns
import tempfile
from google.cloud import storage

def _create_dags_list(dags_directory: str) -> tuple[str, list[str]]:
    if False:
        print('Hello World!')
    temp_dir = tempfile.mkdtemp()
    files_to_ignore = ignore_patterns('__init__.py', '*_test.py')
    copytree(dags_directory, f'{temp_dir}/', ignore=files_to_ignore, dirs_exist_ok=True)
    dags = glob.glob(f'{temp_dir}/*.py')
    return (temp_dir, dags)

def upload_dags_to_composer(dags_directory: str, bucket_name: str, name_replacement: str='dags/') -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a directory, this function moves all DAG files from that directory\n    to a temporary directory, then uploads all contents of the temporary directory\n    to a given cloud storage bucket\n    Args:\n        dags_directory (str): a fully qualified path to a directory that contains a "dags/" subdirectory\n        bucket_name (str): the GCS bucket of the Cloud Composer environment to upload DAGs to\n        name_replacement (str, optional): the name of the "dags/" subdirectory that will be used when constructing the temporary directory path name Defaults to "dags/".\n    '
    (temp_dir, dags) = _create_dags_list(dags_directory)
    if len(dags) > 0:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        for dag in dags:
            dag = dag.replace(f'{temp_dir}/', name_replacement)
            try:
                blob = bucket.blob(dag)
                blob.upload_from_filename(dag)
                print(f'File {dag} uploaded to {bucket_name}/{dag}.')
            except FileNotFoundError:
                current_directory = os.listdir()
                print(f'{name_replacement} directory not found in {current_directory}, you may need to override the default value of name_replacement to point to a relative directory')
                raise
    else:
        print('No DAGs to upload.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dags_directory', help='Relative path to the source directory containing your DAGs')
    parser.add_argument('--dags_bucket', help='Name of the DAGs bucket of your Composer environment without the gs:// prefix')
    args = parser.parse_args()
    upload_dags_to_composer(args.dags_directory, args.dags_bucket)