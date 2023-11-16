from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import redirect
from django.utils.http import is_safe_url

def arg(request):
    if False:
        i = 10
        return i + 15
    return redirect(request.POST.get('next'))

def argh(request):
    if False:
        i = 10
        return i + 15
    return redirect(request.get_host())

def arghh(request):
    if False:
        while True:
            i = 10
    return redirect(request.method)

def argh2(request):
    if False:
        return 10
    url = request.get_host()
    print('something')
    return redirect(url)

def unsafe(request):
    if False:
        return 10
    url = request.headers.get('referrer')
    print('something')
    return redirect(url)

def safe(request):
    if False:
        i = 10
        return i + 15
    url = 'https://lmnop.qrs'
    return redirect(url)

def fine(request):
    if False:
        i = 10
        return i + 15
    return HttpResponseRedirect('https://google.com')

def unsafe2(request):
    if False:
        while True:
            i = 10
    url = request.POST.get('url')
    return HttpResponseRedirect(url)

def legit(request):
    if False:
        return 10
    return HttpResponseRedirect(request.get_full_path())

def url_validation(request):
    if False:
        print('Hello World!')
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