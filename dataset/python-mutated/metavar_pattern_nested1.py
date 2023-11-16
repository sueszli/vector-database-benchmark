from django.shortcuts import redirect

def unsafe1(request):
    if False:
        return 10
    url = request.headers.get('referrer')
    print('something')
    return redirect(url)

def unsafe1(request):
    if False:
        i = 10
        return i + 15
    url = request.get('referrer')
    print('something')
    return redirect(url)

def unsafe2(request):
    if False:
        i = 10
        return i + 15
    url = request.get_full_path('referrer')
    print('something')
    return redirect(url)