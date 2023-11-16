import hug

@hug.get()
def cors_supported(cors: hug.directives.cors='*'):
    if False:
        return 10
    return 'Hello world!'