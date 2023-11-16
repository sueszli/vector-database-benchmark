ROBOT_LISTENER_API_VERSION = 3

def startTest(data, result):
    if False:
        i = 10
        return i + 15
    result.message = '[START] [original] %s [resolved] %s' % (data.name, result.name)

def end_test(data, result):
    if False:
        while True:
            i = 10
    result.message += '\n[END] [original] %s [resolved] %s' % (data.name, result.name)