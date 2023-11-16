import logging
from flask_restful import Resource
logger = logging.getLogger(__name__)

class Photo(Resource):
    """
    Encapsulates a photo resource for an image that is stored in an Amazon Simple
    Storage Service (Amazon S3) bucket.
    """

    def __init__(self, photo_bucket):
        if False:
            while True:
                i = 10
        '\n        :param photo_bucket: The S3 bucket where your photos are stored.\n        '
        self.photo_bucket = photo_bucket

    def get(self, photo_key):
        if False:
            print('Hello World!')
        '\n        Generates a presigned URL that lets you download a photo from your S3 bucket.\n        You can set this URL as the `src` attribute of an HTML `<img>` tag to display\n        the image in an HTML page.\n\n        :param photo_key: The key of the photo object in the S3 bucket.\n        :return: The presigned URL and an HTTP code.\n        '
        url = self.photo_bucket.meta.client.generate_presigned_url(ClientMethod='get_object', Params={'Bucket': self.photo_bucket.name, 'Key': photo_key})
        logger.info('Got presigned URL: %s', url)
        return ({'name': photo_key, 'url': url}, 200)