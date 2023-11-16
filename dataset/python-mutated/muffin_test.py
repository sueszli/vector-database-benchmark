import muffin
app = muffin.Application('web')

@app.register('/text')
def text(request):
    if False:
        return 10
    return 'Hello, World!'