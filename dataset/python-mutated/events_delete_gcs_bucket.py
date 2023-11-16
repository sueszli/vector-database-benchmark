import os
from setup_cleanup import delete_bucket

def delete_bucket_by_name(name: str):
    if False:
        return 10
    if name is None:
        bucket_name = os.getenv('EVENTS_BUCKET_NAME')
        delete_bucket(bucket_name)
    else:
        delete_bucket(name)