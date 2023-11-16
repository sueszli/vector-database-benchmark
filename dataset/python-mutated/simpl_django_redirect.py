from django.shortcuts import redirect
from django.http import HttpResponseRedirect, HttpResponse
from django.utils.http import is_safe_url

def unsafe(request):
    if False:
        for i in range(10):
            print('nop')
    url = request.headers.get('referrer')
    print('something')
    return redirect(url)

def safe(request):
    if False:
        print('Hello World!')
    url = 'https://lmnop.qrs'
    return redirect(url)

def unsafe2(request):
    if False:
        for i in range(10):
            print('nop')
    url = request.POST.get('url')
    return HttpResponseRedirect(url)

def unsafe3(request):
    if False:
        return 10
    url = request.POST['url']
    return HttpResponseRedirect(url)

def fine(request):
    if False:
        for i in range(10):
            print('nop')
    return HttpResponseRedirect(request.get_full_path())

def url_validation(request):
    if False:
        while True:
            i = 10
    next = request.POST.get('next', request.GET.get('next'))
    if (next or not request.is_ajax()) and (not is_safe_url(url=next, allowed_hosts=request.get_host())):
        next = '/index'
    response = HttpResponseRedirect(next) if next else HttpResponse(status=204)
    return response

def url_validation2(request):
    if False:
        print('Hello World!')
    next = request.POST.get('next', request.GET.get('next'))
    ok = is_safe_url(url=next, allowed_hosts=request.get_host())
    if ok:
        response = HttpResponseRedirect(next) if next else HttpResponse(status=204)
    else:
        response = HttpResponseRedirect('index')
    return response