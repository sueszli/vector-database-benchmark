ROBOT_LISTENER_API_VERSION = 2

def startTest(name, info):
    if False:
        for i in range(10):
            print('nop')
    print('[START] [original] %s [resolved] %s' % (info['originalname'], name))

def end_test(name, info):
    if False:
        for i in range(10):
            print('nop')
    print('[END] [original] %s [resolved] %s' % (info['originalname'], name))