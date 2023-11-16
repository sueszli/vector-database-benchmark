from django.shortcuts import render

def test_view1(request):
    if False:
        i = 10
        return i + 15
    return render(request, 'index.html', locals())

def test_view2(request):
    if False:
        for i in range(10):
            print('nop')
    return render(request, 'index.html', context=locals())

def test_view3(request):
    if False:
        i = 10
        return i + 15
    return render(request, 'index.html')

def test_view4(request):
    if False:
        for i in range(10):
            print('nop')
    return render(request, 'index.html', {})

def test_view5(request):
    if False:
        i = 10
        return i + 15
    return render(request, 'index.html', context={})