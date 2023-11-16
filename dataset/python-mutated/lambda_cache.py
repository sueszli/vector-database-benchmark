counter = 0

def handler(event, context):
    if False:
        i = 10
        return i + 15
    global counter
    result = {'counter': counter}
    counter += 1
    return result