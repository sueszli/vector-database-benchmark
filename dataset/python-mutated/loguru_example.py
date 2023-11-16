from loguru import logger

def main():
    if False:
        print('Hello World!')
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
if __name__ == '__main__':
    main()