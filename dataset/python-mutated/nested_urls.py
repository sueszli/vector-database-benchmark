from django.urls import include, path
from django.views import View

def view1(request):
    if False:
        print('Hello World!')
    pass

def view2(request):
    if False:
        return 10
    pass

class View3(View):
    pass
nested = ([path('view1/', view1, name='view1'), path('view3/', View3.as_view(), name='view3')], 'backend')
urlpatterns = [path('some/path/', include(nested, namespace='nested')), path('view2/', view2, name='view2')]