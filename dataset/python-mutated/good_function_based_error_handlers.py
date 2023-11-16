urlpatterns = []
handler400 = __name__ + '.good_handler'
handler403 = __name__ + '.good_handler'
handler404 = __name__ + '.good_handler'
handler500 = __name__ + '.good_handler'

def good_handler(request, exception=None, foo='bar'):
    if False:
        while True:
            i = 10
    pass