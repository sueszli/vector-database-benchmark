"""
Response selection methods determines which response should be used in
the event that multiple responses are generated within a logic adapter.
"""
import logging

def get_most_frequent_response(input_statement, response_list, storage=None):
    if False:
        i = 10
        return i + 15
    '\n    :param input_statement: A statement, that closely matches an input to the chat bot.\n    :type input_statement: Statement\n\n    :param response_list: A list of statement options to choose a response from.\n    :type response_list: list\n\n    :param storage: An instance of a storage adapter to allow the response selection\n                    method to access other statements if needed.\n    :type storage: StorageAdapter\n\n    :return: The response statement with the greatest number of occurrences.\n    :rtype: Statement\n    '
    matching_response = None
    occurrence_count = -1
    logger = logging.getLogger(__name__)
    logger.info('Selecting response with greatest number of occurrences.')
    for statement in response_list:
        count = len(list(storage.filter(text=statement.text, in_response_to=input_statement.text)))
        if count >= occurrence_count:
            matching_response = statement
            occurrence_count = count
    return matching_response

def get_first_response(input_statement, response_list, storage=None):
    if False:
        return 10
    '\n    :param input_statement: A statement, that closely matches an input to the chat bot.\n    :type input_statement: Statement\n\n    :param response_list: A list of statement options to choose a response from.\n    :type response_list: list\n\n    :param storage: An instance of a storage adapter to allow the response selection\n                    method to access other statements if needed.\n    :type storage: StorageAdapter\n\n    :return: Return the first statement in the response list.\n    :rtype: Statement\n    '
    logger = logging.getLogger(__name__)
    logger.info('Selecting first response from list of {} options.'.format(len(response_list)))
    return response_list[0]

def get_random_response(input_statement, response_list, storage=None):
    if False:
        return 10
    '\n    :param input_statement: A statement, that closely matches an input to the chat bot.\n    :type input_statement: Statement\n\n    :param response_list: A list of statement options to choose a response from.\n    :type response_list: list\n\n    :param storage: An instance of a storage adapter to allow the response selection\n                    method to access other statements if needed.\n    :type storage: StorageAdapter\n\n    :return: Choose a random response from the selection.\n    :rtype: Statement\n    '
    from random import choice
    logger = logging.getLogger(__name__)
    logger.info('Selecting a response from list of {} options.'.format(len(response_list)))
    return choice(response_list)