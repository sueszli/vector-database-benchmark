import hug

@hug.get('/text', output_format=hug.output_format.text, parse_body=False)
def text():
    if False:
        i = 10
        return i + 15
    return 'Hello, World!'
app = hug.API(__name__).http.server()