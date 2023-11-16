"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Rekognition to
recognize people, objects, and text in images.

The usage demo in this file uses images in the .media folder. If you run this code
without cloning the GitHub repository, you must first download the image files from
    https://github.com/awsdocs/aws-doc-sdk-examples/tree/master/python/example_code/rekognition/.media
"""
import logging
from pprint import pprint
import boto3
from botocore.exceptions import ClientError
import requests
from rekognition_objects import RekognitionFace, RekognitionCelebrity, RekognitionLabel, RekognitionModerationLabel, RekognitionText, show_bounding_boxes, show_polygons
logger = logging.getLogger(__name__)

class RekognitionImage:
    """
    Encapsulates an Amazon Rekognition image. This class is a thin wrapper
    around parts of the Boto3 Amazon Rekognition API.
    """

    def __init__(self, image, image_name, rekognition_client):
        if False:
            print('Hello World!')
        '\n        Initializes the image object.\n\n        :param image: Data that defines the image, either the image bytes or\n                      an Amazon S3 bucket and object key.\n        :param image_name: The name of the image.\n        :param rekognition_client: A Boto3 Rekognition client.\n        '
        self.image = image
        self.image_name = image_name
        self.rekognition_client = rekognition_client

    @classmethod
    def from_file(cls, image_file_name, rekognition_client, image_name=None):
        if False:
            return 10
        '\n        Creates a RekognitionImage object from a local file.\n\n        :param image_file_name: The file name of the image. The file is opened and its\n                                bytes are read.\n        :param rekognition_client: A Boto3 Rekognition client.\n        :param image_name: The name of the image. If this is not specified, the\n                           file name is used as the image name.\n        :return: The RekognitionImage object, initialized with image bytes from the\n                 file.\n        '
        with open(image_file_name, 'rb') as img_file:
            image = {'Bytes': img_file.read()}
        name = image_file_name if image_name is None else image_name
        return cls(image, name, rekognition_client)

    @classmethod
    def from_bucket(cls, s3_object, rekognition_client):
        if False:
            print('Hello World!')
        '\n        Creates a RekognitionImage object from an Amazon S3 object.\n\n        :param s3_object: An Amazon S3 object that identifies the image. The image\n                          is not retrieved until needed for a later call.\n        :param rekognition_client: A Boto3 Rekognition client.\n        :return: The RekognitionImage object, initialized with Amazon S3 object data.\n        '
        image = {'S3Object': {'Bucket': s3_object.bucket_name, 'Name': s3_object.key}}
        return cls(image, s3_object.key, rekognition_client)

    def detect_faces(self):
        if False:
            i = 10
            return i + 15
        '\n        Detects faces in the image.\n\n        :return: The list of faces found in the image.\n        '
        try:
            response = self.rekognition_client.detect_faces(Image=self.image, Attributes=['ALL'])
            faces = [RekognitionFace(face) for face in response['FaceDetails']]
            logger.info('Detected %s faces.', len(faces))
        except ClientError:
            logger.exception("Couldn't detect faces in %s.", self.image_name)
            raise
        else:
            return faces

    def compare_faces(self, target_image, similarity):
        if False:
            print('Hello World!')
        '\n        Compares faces in the image with the largest face in the target image.\n\n        :param target_image: The target image to compare against.\n        :param similarity: Faces in the image must have a similarity value greater\n                           than this value to be included in the results.\n        :return: A tuple. The first element is the list of faces that match the\n                 reference image. The second element is the list of faces that have\n                 a similarity value below the specified threshold.\n        '
        try:
            response = self.rekognition_client.compare_faces(SourceImage=self.image, TargetImage=target_image.image, SimilarityThreshold=similarity)
            matches = [RekognitionFace(match['Face']) for match in response['FaceMatches']]
            unmatches = [RekognitionFace(face) for face in response['UnmatchedFaces']]
            logger.info('Found %s matched faces and %s unmatched faces.', len(matches), len(unmatches))
        except ClientError:
            logger.exception("Couldn't match faces from %s to %s.", self.image_name, target_image.image_name)
            raise
        else:
            return (matches, unmatches)

    def detect_labels(self, max_labels):
        if False:
            print('Hello World!')
        '\n        Detects labels in the image. Labels are objects and people.\n\n        :param max_labels: The maximum number of labels to return.\n        :return: The list of labels detected in the image.\n        '
        try:
            response = self.rekognition_client.detect_labels(Image=self.image, MaxLabels=max_labels)
            labels = [RekognitionLabel(label) for label in response['Labels']]
            logger.info('Found %s labels in %s.', len(labels), self.image_name)
        except ClientError:
            logger.info("Couldn't detect labels in %s.", self.image_name)
            raise
        else:
            return labels

    def detect_moderation_labels(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Detects moderation labels in the image. Moderation labels identify content\n        that may be inappropriate for some audiences.\n\n        :return: The list of moderation labels found in the image.\n        '
        try:
            response = self.rekognition_client.detect_moderation_labels(Image=self.image)
            labels = [RekognitionModerationLabel(label) for label in response['ModerationLabels']]
            logger.info('Found %s moderation labels in %s.', len(labels), self.image_name)
        except ClientError:
            logger.exception("Couldn't detect moderation labels in %s.", self.image_name)
            raise
        else:
            return labels

    def detect_text(self):
        if False:
            return 10
        '\n        Detects text in the image.\n\n        :return The list of text elements found in the image.\n        '
        try:
            response = self.rekognition_client.detect_text(Image=self.image)
            texts = [RekognitionText(text) for text in response['TextDetections']]
            logger.info('Found %s texts in %s.', len(texts), self.image_name)
        except ClientError:
            logger.exception("Couldn't detect text in %s.", self.image_name)
            raise
        else:
            return texts

    def recognize_celebrities(self):
        if False:
            return 10
        '\n        Detects celebrities in the image.\n\n        :return: A tuple. The first element is the list of celebrities found in\n                 the image. The second element is the list of faces that were\n                 detected but did not match any known celebrities.\n        '
        try:
            response = self.rekognition_client.recognize_celebrities(Image=self.image)
            celebrities = [RekognitionCelebrity(celeb) for celeb in response['CelebrityFaces']]
            other_faces = [RekognitionFace(face) for face in response['UnrecognizedFaces']]
            logger.info('Found %s celebrities and %s other faces in %s.', len(celebrities), len(other_faces), self.image_name)
        except ClientError:
            logger.exception("Couldn't detect celebrities in %s.", self.image_name)
            raise
        else:
            return (celebrities, other_faces)

def usage_demo():
    if False:
        for i in range(10):
            print('nop')
    print('-' * 88)
    print('Welcome to the Amazon Rekognition image detection demo!')
    print('-' * 88)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    rekognition_client = boto3.client('rekognition')
    street_scene_file_name = '.media/pexels-kaique-rocha-109919.jpg'
    celebrity_file_name = '.media/pexels-pixabay-53370.jpg'
    one_girl_url = 'https://dhei5unw3vrsx.cloudfront.net/images/source3_resized.jpg'
    three_girls_url = 'https://dhei5unw3vrsx.cloudfront.net/images/target3_resized.jpg'
    swimwear_object = boto3.resource('s3').Object('console-sample-images-pdx', 'yoga_swimwear.jpg')
    book_file_name = '.media/pexels-christina-morillo-1181671.jpg'
    street_scene_image = RekognitionImage.from_file(street_scene_file_name, rekognition_client)
    print(f'Detecting faces in {street_scene_image.image_name}...')
    faces = street_scene_image.detect_faces()
    print(f'Found {len(faces)} faces, here are the first three.')
    for face in faces[:3]:
        pprint(face.to_dict())
    show_bounding_boxes(street_scene_image.image['Bytes'], [[face.bounding_box for face in faces]], ['aqua'])
    input('Press Enter to continue.')
    print(f'Detecting labels in {street_scene_image.image_name}...')
    labels = street_scene_image.detect_labels(100)
    print(f'Found {len(labels)} labels.')
    for label in labels:
        pprint(label.to_dict())
    names = []
    box_sets = []
    colors = ['aqua', 'red', 'white', 'blue', 'yellow', 'green']
    for label in labels:
        if label.instances:
            names.append(label.name)
            box_sets.append([inst['BoundingBox'] for inst in label.instances])
    print(f'Showing bounding boxes for {names} in {colors[:len(names)]}.')
    show_bounding_boxes(street_scene_image.image['Bytes'], box_sets, colors[:len(names)])
    input('Press Enter to continue.')
    celebrity_image = RekognitionImage.from_file(celebrity_file_name, rekognition_client)
    print(f'Detecting celebrities in {celebrity_image.image_name}...')
    (celebs, others) = celebrity_image.recognize_celebrities()
    print(f'Found {len(celebs)} celebrities.')
    for celeb in celebs:
        pprint(celeb.to_dict())
    show_bounding_boxes(celebrity_image.image['Bytes'], [[celeb.face.bounding_box for celeb in celebs]], ['aqua'])
    input('Press Enter to continue.')
    girl_image_response = requests.get(one_girl_url)
    girl_image = RekognitionImage({'Bytes': girl_image_response.content}, 'one-girl', rekognition_client)
    group_image_response = requests.get(three_girls_url)
    group_image = RekognitionImage({'Bytes': group_image_response.content}, 'three-girls', rekognition_client)
    print('Comparing reference face to group of faces...')
    (matches, unmatches) = girl_image.compare_faces(group_image, 80)
    print(f'Found {len(matches)} face matching the reference face.')
    show_bounding_boxes(group_image.image['Bytes'], [[match.bounding_box for match in matches]], ['aqua'])
    input('Press Enter to continue.')
    swimwear_image = RekognitionImage.from_bucket(swimwear_object, rekognition_client)
    print(f'Detecting suggestive content in {swimwear_object.key}...')
    labels = swimwear_image.detect_moderation_labels()
    print(f'Found {len(labels)} moderation labels.')
    for label in labels:
        pprint(label.to_dict())
    input('Press Enter to continue.')
    book_image = RekognitionImage.from_file(book_file_name, rekognition_client)
    print(f'Detecting text in {book_image.image_name}...')
    texts = book_image.detect_text()
    print(f'Found {len(texts)} text instances. Here are the first seven:')
    for text in texts[:7]:
        pprint(text.to_dict())
    show_polygons(book_image.image['Bytes'], [text.geometry['Polygon'] for text in texts], 'aqua')
    print('Thanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    usage_demo()