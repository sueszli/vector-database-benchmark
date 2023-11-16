from wal_e import exception
from wal_e import retries
from wal_e.worker.base import _Deleter

def _on_error(blob):
    if False:
        return 10
    pass

class Deleter(_Deleter):

    @retries.retry()
    def _delete_batch(self, page):
        if False:
            i = 10
            return i + 15
        bucket_name = page[0].bucket.name
        for blob in page:
            if blob.bucket.name != bucket_name:
                raise exception.UserCritical(msg='submitted blobs are not part of the same bucket', detail='The clashing bucket names are {0} and {1}.'.format(blob.bucket.name, bucket_name), hint='This should be reported as a bug.')
        bucket = page[0].bucket
        bucket.delete_blobs(page, on_error=_on_error)