from django.http import HttpRequest, HttpResponse

def test_to_response(request: HttpRequest):
    if False:
        return 10
    source = request.GET['bad']
    return HttpResponse(content=source)