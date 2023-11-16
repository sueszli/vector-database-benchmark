from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import redirect
from django.utils.http import is_safe_url

def unsafe(request):
    if False:
        i = 10
        return i + 15
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
        for i in range(10):
            print('nop')
    url = request.POST['url']
    return HttpResponseRedirect(url)

def unsafe4(request):
    if False:
        i = 10
        return i + 15
    url = request.get_referrer()
    if url:
        return HttpResponseRedirect(url)

def fine(request):
    if False:
        while True:
            i = 10
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
        while True:
            i = 10
    next = request.POST.get('next', request.GET.get('next'))
    ok = is_safe_url(url=next, allowed_hosts=request.get_host())
    if ok:
        response = HttpResponseRedirect(next) if next else HttpResponse(status=204)
    else:
        response = HttpResponseRedirect('index')
    return response