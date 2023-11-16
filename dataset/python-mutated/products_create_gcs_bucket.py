import datetime
from pathlib import Path, PurePath
import google.auth
from setup_cleanup import create_bucket, upload_blob
project_id = google.auth.default()[1]
timestamp_ = datetime.datetime.now().timestamp().__round__()
BUCKET_NAME = f'{project_id}_products_{timestamp_}'

def create_gcs_bucket(bucket_name=BUCKET_NAME):
    if False:
        print('Hello World!')
    create_bucket(bucket_name)
    path_to_resources_dir = PurePath(Path.home(), 'cloudshell_open/python-docs-samples/retail/interactive-tutorials/resources')
    upload_blob(bucket_name, str(path_to_resources_dir / 'products.json'))
    upload_blob(bucket_name, str(path_to_resources_dir / 'products_some_invalid.json'))
    print(f'\nThe gcs bucket {bucket_name} was created')
if __name__ == '__main__':
    create_gcs_bucket()