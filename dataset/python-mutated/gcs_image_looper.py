"""This executable loops image filepaths from a gcs bucket file."""
import random
import time
from google.api_core.exceptions import AlreadyExists
from google.cloud import pubsub_v1
from google.cloud import storage
project_id = 'apache-beam-testing'
gcs_bucket = 'apache-beam-ml'
num_images_per_second = 5
publisher = pubsub_v1.PublisherClient()
image_file_path = 'testing/inputs/openimage_50k_benchmark.txt'
topic_name = 'Imagenet_openimage_50k_benchmark'
topic_path = publisher.topic_path(project_id, topic_name)

class ImageLooper(object):
    """Loop the images in a gcs bucket file and publish them to a pubsub topic.
  """
    content = ''
    cursor = 0

    def __init__(self, filename):
        if False:
            i = 10
            return i + 15
        self._read_gcs_file(filename)

    def get_next_image(self):
        if False:
            while True:
                i = 10
        'Returns the next image randomly.'
        next_image = ''
        while not next_image:
            image_id = random.randint(0, len(self.content) - 1)
            next_image = self.content[image_id]
        return next_image

    def _read_gcs_file(self, filename):
        if False:
            i = 10
            return i + 15
        client = storage.Client()
        bucket = client.get_bucket(gcs_bucket)
        blob = bucket.get_blob(filename)
        self.content = blob.download_as_string().decode('utf-8').split('\n')
try:
    publisher.create_topic(request={'name': topic_path})
except AlreadyExists:
    pass
looper = ImageLooper(image_file_path)
while True:
    image = looper.get_next_image()
    publisher.publish(topic_path, data=image.encode('utf-8'))
    time.sleep(1 / num_images_per_second)