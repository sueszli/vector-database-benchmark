import bobo

@bobo.query('/text', content_type='text/plain')
def text():
    if False:
        print('Hello World!')
    return 'Hello, world!'
app = bobo.Application(bobo_resources=__name__)