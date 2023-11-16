"""
Purpose

Shows how to implement an AWS Lambda function that uses machine learning.
"""
import logging
from threading import Timer
import numpy
import greengrass_machine_learning_sdk as gg_ml
with open('/test_img/test.jpg', 'rb') as f:
    content = f.read()
client = gg_ml.client('inference')

def infer():
    if False:
        i = 10
        return i + 15
    logging.info('Invoking Greengrass ML Inference service')
    try:
        resp = client.invoke_inference_service(AlgoType='image-classification', ServiceName='imageClassification', ContentType='image/jpeg', Body=content)
    except gg_ml.GreengrassInferenceException as e:
        logging.info('Inference exception %s("%s")', e.__class__.__name__, e)
        return
    except gg_ml.GreengrassDependencyException as e:
        logging.info('Dependency exception %s("%s")', e.__class__.__name__, e)
        return
    logging.info('Response: %s', resp)
    predictions = resp['Body'].read()
    logging.info('Predictions: %s', predictions)
    predictions = predictions[1:-1]
    predictions_arr = numpy.fromstring(predictions, sep=',')
    logging.info('Split into %s predictions.', len(predictions_arr))
    Timer(1, infer).start()
infer()

def function_handler(event, context):
    if False:
        return 10
    return