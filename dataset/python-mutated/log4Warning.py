import logging
from typing import List
logger = logging.getLogger(__name__)
warning_messages: List[str] = []

def output_suggestions():
    if False:
        return 10
    global warning_messages
    if len(warning_messages) > 0:
        logger.warning(f'\n*****************Nano performance Suggestions*****************')
        for message in warning_messages:
            logger.warning(message)
        logger.warning(f'\n*****************Nano performance Suggestions*****************')

def register_suggestion(warning_message: str):
    if False:
        i = 10
        return i + 15
    global warning_messages
    print(warning_message, flush=True)
    warning_messages.append(warning_message)