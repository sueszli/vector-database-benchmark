import spam

@spam.eggs
def eggs_and_sausage(*args):
    if False:
        for i in range(10):
            print('nop')
    return 'spam'