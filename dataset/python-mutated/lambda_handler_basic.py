"""
Purpose

Shows how to implement an AWS Lambda function that handles input from direct
invocation.
"""
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    if False:
        return 10
    "\n    Accepts an action and a single number, performs the specified action on the number,\n    and returns the result. The only allowable action is 'increment'.\n\n    :param event: The event dict that contains the parameters sent when the function\n                  is invoked.\n    :param context: The context in which the function is called.\n    :return: The result of the action.\n    "
    result = None
    action = event.get('action')
    if action == 'increment':
        result = event.get('number', 0) + 1
        logger.info('Calculated result of %s', result)
    else:
        logger.error('%s is not a valid action.', action)
    response = {'result': result}
    return response