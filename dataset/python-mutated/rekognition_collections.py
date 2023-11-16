"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Rekognition to
create a collection that contains faces indexed from a series of images. The
collection is then searched for faces that match a reference face.

The usage demo in this file uses images in the .media folder. If you run this code
without cloning the GitHub repository, you must first download the image files from
    https://github.com/awsdocs/aws-doc-sdk-examples/tree/master/python/example_code/rekognition/.media
"""
import logging
from pprint import pprint
import boto3
from botocore.exceptions import ClientError
from rekognition_objects import RekognitionFace
from rekognition_image_detection import RekognitionImage
logger = logging.getLogger(__name__)

class RekognitionCollection:
    """
    Encapsulates an Amazon Rekognition collection. This class is a thin wrapper
    around parts of the Boto3 Amazon Rekognition API.
    """

    def __init__(self, collection, rekognition_client):
        if False:
            return 10
        '\n        Initializes a collection object.\n\n        :param collection: Collection data in the format returned by a call to\n                           create_collection.\n        :param rekognition_client: A Boto3 Rekognition client.\n        '
        self.collection_id = collection['CollectionId']
        (self.collection_arn, self.face_count, self.created) = self._unpack_collection(collection)
        self.rekognition_client = rekognition_client

    @staticmethod
    def _unpack_collection(collection):
        if False:
            return 10
        '\n        Unpacks optional parts of a collection that can be returned by\n        describe_collection.\n\n        :param collection: The collection data.\n        :return: A tuple of the data in the collection.\n        '
        return (collection.get('CollectionArn'), collection.get('FaceCount', 0), collection.get('CreationTimestamp'))

    def to_dict(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Renders parts of the collection data to a dict.\n\n        :return: The collection data as a dict.\n        '
        rendering = {'collection_id': self.collection_id, 'collection_arn': self.collection_arn, 'face_count': self.face_count, 'created': self.created}
        return rendering

    def describe_collection(self):
        if False:
            while True:
                i = 10
        '\n        Gets data about the collection from the Amazon Rekognition service.\n\n        :return: The collection rendered as a dict.\n        '
        try:
            response = self.rekognition_client.describe_collection(CollectionId=self.collection_id)
            response['CollectionArn'] = response.get('CollectionARN')
            (self.collection_arn, self.face_count, self.created) = self._unpack_collection(response)
            logger.info('Got data for collection %s.', self.collection_id)
        except ClientError:
            logger.exception("Couldn't get data for collection %s.", self.collection_id)
            raise
        else:
            return self.to_dict()

    def delete_collection(self):
        if False:
            while True:
                i = 10
        '\n        Deletes the collection.\n        '
        try:
            self.rekognition_client.delete_collection(CollectionId=self.collection_id)
            logger.info('Deleted collection %s.', self.collection_id)
            self.collection_id = None
        except ClientError:
            logger.exception("Couldn't delete collection %s.", self.collection_id)
            raise

    def index_faces(self, image, max_faces):
        if False:
            return 10
        "\n        Finds faces in the specified image, indexes them, and stores them in the\n        collection.\n\n        :param image: The image to index.\n        :param max_faces: The maximum number of faces to index.\n        :return: A tuple. The first element is a list of indexed faces.\n                 The second element is a list of faces that couldn't be indexed.\n        "
        try:
            response = self.rekognition_client.index_faces(CollectionId=self.collection_id, Image=image.image, ExternalImageId=image.image_name, MaxFaces=max_faces, DetectionAttributes=['ALL'])
            indexed_faces = [RekognitionFace({**face['Face'], **face['FaceDetail']}) for face in response['FaceRecords']]
            unindexed_faces = [RekognitionFace(face['FaceDetail']) for face in response['UnindexedFaces']]
            logger.info('Indexed %s faces in %s. Could not index %s faces.', len(indexed_faces), image.image_name, len(unindexed_faces))
        except ClientError:
            logger.exception("Couldn't index faces in image %s.", image.image_name)
            raise
        else:
            return (indexed_faces, unindexed_faces)

    def list_faces(self, max_results):
        if False:
            while True:
                i = 10
        '\n        Lists the faces currently indexed in the collection.\n\n        :param max_results: The maximum number of faces to return.\n        :return: The list of faces in the collection.\n        '
        try:
            response = self.rekognition_client.list_faces(CollectionId=self.collection_id, MaxResults=max_results)
            faces = [RekognitionFace(face) for face in response['Faces']]
            logger.info('Found %s faces in collection %s.', len(faces), self.collection_id)
        except ClientError:
            logger.exception("Couldn't list faces in collection %s.", self.collection_id)
            raise
        else:
            return faces

    def search_faces_by_image(self, image, threshold, max_faces):
        if False:
            for i in range(10):
                print('nop')
        '\n        Searches for faces in the collection that match the largest face in the\n        reference image.\n\n        :param image: The image that contains the reference face to search for.\n        :param threshold: The match confidence must be greater than this value\n                          for a face to be included in the results.\n        :param max_faces: The maximum number of faces to return.\n        :return: A tuple. The first element is the face found in the reference image.\n                 The second element is the list of matching faces found in the\n                 collection.\n        '
        try:
            response = self.rekognition_client.search_faces_by_image(CollectionId=self.collection_id, Image=image.image, FaceMatchThreshold=threshold, MaxFaces=max_faces)
            image_face = RekognitionFace({'BoundingBox': response['SearchedFaceBoundingBox'], 'Confidence': response['SearchedFaceConfidence']})
            collection_faces = [RekognitionFace(face['Face']) for face in response['FaceMatches']]
            logger.info('Found %s faces in the collection that match the largest face in %s.', len(collection_faces), image.image_name)
        except ClientError:
            logger.exception("Couldn't search for faces in %s that match %s.", self.collection_id, image.image_name)
            raise
        else:
            return (image_face, collection_faces)

    def search_faces(self, face_id, threshold, max_faces):
        if False:
            for i in range(10):
                print('nop')
        '\n        Searches for faces in the collection that match another face from the\n        collection.\n\n        :param face_id: The ID of the face in the collection to search for.\n        :param threshold: The match confidence must be greater than this value\n                          for a face to be included in the results.\n        :param max_faces: The maximum number of faces to return.\n        :return: The list of matching faces found in the collection. This list does\n                 not contain the face specified by `face_id`.\n        '
        try:
            response = self.rekognition_client.search_faces(CollectionId=self.collection_id, FaceId=face_id, FaceMatchThreshold=threshold, MaxFaces=max_faces)
            faces = [RekognitionFace(face['Face']) for face in response['FaceMatches']]
            logger.info('Found %s faces in %s that match %s.', len(faces), self.collection_id, face_id)
        except ClientError:
            logger.exception("Couldn't search for faces in %s that match %s.", self.collection_id, face_id)
            raise
        else:
            return faces

    def delete_faces(self, face_ids):
        if False:
            while True:
                i = 10
        '\n        Deletes faces from the collection.\n\n        :param face_ids: The list of IDs of faces to delete.\n        :return: The list of IDs of faces that were deleted.\n        '
        try:
            response = self.rekognition_client.delete_faces(CollectionId=self.collection_id, FaceIds=face_ids)
            deleted_ids = response['DeletedFaces']
            logger.info('Deleted %s faces from %s.', len(deleted_ids), self.collection_id)
        except ClientError:
            logger.exception("Couldn't delete faces from %s.", self.collection_id)
            raise
        else:
            return deleted_ids

class RekognitionCollectionManager:
    """
    Encapsulates Amazon Rekognition collection management functions.
    This class is a thin wrapper around parts of the Boto3 Amazon Rekognition API.
    """

    def __init__(self, rekognition_client):
        if False:
            i = 10
            return i + 15
        '\n        Initializes the collection manager object.\n\n        :param rekognition_client: A Boto3 Rekognition client.\n        '
        self.rekognition_client = rekognition_client

    def create_collection(self, collection_id):
        if False:
            return 10
        '\n        Creates an empty collection.\n\n        :param collection_id: Text that identifies the collection.\n        :return: The newly created collection.\n        '
        try:
            response = self.rekognition_client.create_collection(CollectionId=collection_id)
            response['CollectionId'] = collection_id
            collection = RekognitionCollection(response, self.rekognition_client)
            logger.info('Created collection %s.', collection_id)
        except ClientError:
            logger.exception("Couldn't create collection %s.", collection_id)
            raise
        else:
            return collection

    def list_collections(self, max_results):
        if False:
            while True:
                i = 10
        '\n        Lists collections for the current account.\n\n        :param max_results: The maximum number of collections to return.\n        :return: The list of collections for the current account.\n        '
        try:
            response = self.rekognition_client.list_collections(MaxResults=max_results)
            collections = [RekognitionCollection({'CollectionId': col_id}, self.rekognition_client) for col_id in response['CollectionIds']]
        except ClientError:
            logger.exception("Couldn't list collections.")
            raise
        else:
            return collections

def usage_demo():
    if False:
        return 10
    print('-' * 88)
    print('Welcome to the Amazon Rekognition face collection demo!')
    print('-' * 88)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    rekognition_client = boto3.client('rekognition')
    images = [RekognitionImage.from_file('.media/pexels-agung-pandit-wiguna-1128316.jpg', rekognition_client, image_name='sitting'), RekognitionImage.from_file('.media/pexels-agung-pandit-wiguna-1128317.jpg', rekognition_client, image_name='hopping'), RekognitionImage.from_file('.media/pexels-agung-pandit-wiguna-1128318.jpg', rekognition_client, image_name='biking')]
    collection_mgr = RekognitionCollectionManager(rekognition_client)
    collection = collection_mgr.create_collection('doc-example-collection-demo')
    print(f'Created collection {collection.collection_id}:')
    pprint(collection.describe_collection())
    print('Indexing faces from three images:')
    for image in images:
        collection.index_faces(image, 10)
    print('Listing faces in collection:')
    faces = collection.list_faces(10)
    for face in faces:
        pprint(face.to_dict())
    input('Press Enter to continue.')
    print(f'Searching for faces in the collection that match the first face in the list (Face ID: {faces[0].face_id}.')
    found_faces = collection.search_faces(faces[0].face_id, 80, 10)
    print(f'Found {len(found_faces)} matching faces.')
    for face in found_faces:
        pprint(face.to_dict())
    input('Press Enter to continue.')
    print(f'Searching for faces in the collection that match the largest face in {images[0].image_name}.')
    (image_face, match_faces) = collection.search_faces_by_image(images[0], 80, 10)
    print(f'The largest face in {images[0].image_name} is:')
    pprint(image_face.to_dict())
    print(f'Found {len(match_faces)} matching faces.')
    for face in match_faces:
        pprint(face.to_dict())
    input('Press Enter to continue.')
    collection.delete_collection()
    print('Thanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    usage_demo()