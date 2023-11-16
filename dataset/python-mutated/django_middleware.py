from google.cloud import ndb

def ndb_django_middleware(get_response):
    if False:
        i = 10
        return i + 15
    client = ndb.Client()

    def middleware(request):
        if False:
            return 10
        with client.context():
            return get_response(request)
    return middleware