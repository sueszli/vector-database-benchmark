def main_function():
    if False:
        while True:
            i = 10
    try:
        process()
        handle()
        finish()
    except Exception as ex:
        logger.exception(f'Found an error: {ex}')

def main_function():
    if False:
        return 10
    try:
        process()
        handle()
        finish()
    except ValueError as bad:
        if True is False:
            for i in range(10):
                logger.exception(f'Found an error: {bad} {good}')
    except IndexError as bad:
        logger.exception(f'Found an error: {bad} {bad}')
    except Exception as bad:
        logger.exception(f'Found an error: {bad}')
        logger.exception(f'Found an error: {bad}')
        if True:
            logger.exception(f'Found an error: {bad}')
import logging
logger = logging.getLogger(__name__)

def func_fstr():
    if False:
        while True:
            i = 10
    try:
        ...
    except Exception as ex:
        logger.exception(f'Logging an error: {ex}')

def func_concat():
    if False:
        print('Hello World!')
    try:
        ...
    except Exception as ex:
        logger.exception('Logging an error: ' + str(ex))

def func_comma():
    if False:
        return 10
    try:
        ...
    except Exception as ex:
        logger.exception('Logging an error:', ex)

def main_function():
    if False:
        return 10
    try:
        process()
        handle()
        finish()
    except Exception as ex:
        logger.exception(f'Found an error: {er}')
        logger.exception(f'Found an error: {ex.field}')
from logging import error, exception

def main_function():
    if False:
        print('Hello World!')
    try:
        process()
        handle()
        finish()
    except Exception as ex:
        exception(f'Found an error: {ex}')

def main_function():
    if False:
        return 10
    try:
        process()
        handle()
        finish()
    except ValueError as bad:
        if True is False:
            for i in range(10):
                exception(f'Found an error: {bad} {good}')
    except IndexError as bad:
        exception(f'Found an error: {bad} {bad}')
    except Exception as bad:
        exception(f'Found an error: {bad}')
        exception(f'Found an error: {bad}')
        if True:
            exception(f'Found an error: {bad}')

def func_fstr():
    if False:
        i = 10
        return i + 15
    try:
        ...
    except Exception as ex:
        exception(f'Logging an error: {ex}')

def func_concat():
    if False:
        for i in range(10):
            print('nop')
    try:
        ...
    except Exception as ex:
        exception('Logging an error: ' + str(ex))

def func_comma():
    if False:
        return 10
    try:
        ...
    except Exception as ex:
        exception('Logging an error:', ex)

def main_function():
    if False:
        i = 10
        return i + 15
    try:
        process()
        handle()
        finish()
    except Exception as ex:
        exception(f'Found an error: {er}')
        exception(f'Found an error: {ex.field}')