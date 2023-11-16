from google.cloud import ndb
import django_middleware

def test_ndb_django_middleware():
    if False:
        print('Hello World!')

    def fake_get_response(request):
        if False:
            return 10
        ndb.context.get_context()
    wrapped_function = django_middleware.ndb_django_middleware(fake_get_response)
    wrapped_function(None)