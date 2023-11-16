import faust

def create_app():
    if False:
        while True:
            i = 10
    return faust.App('proj323', origin='proj323', autodiscover=True)
app = create_app()