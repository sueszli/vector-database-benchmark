from wal_e import exception
from wal_e import retries
from wal_e.worker.base import _Deleter

class Deleter(_Deleter):

    @retries.retry()
    def _delete_batch(self, page):
        if False:
            for i in range(10):
                print('nop')
        bucket_name = page[0].bucket.name
        for key in page:
            if key.bucket.name != bucket_name:
                raise exception.UserCritical(msg='submitted keys are not part of the same bucket', detail='The clashing bucket names are {0} and {1}.'.format(key.bucket.name, bucket_name), hint='This should be reported as a bug.')
        bucket = page[0].bucket
        bucket.delete_keys([key.name for key in page])