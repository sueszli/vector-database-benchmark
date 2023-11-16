def foo():
    if False:
        for i in range(10):
            print('nop')
    logging.info(f'Current bento version is {current_version}, latest is {latest_version}')