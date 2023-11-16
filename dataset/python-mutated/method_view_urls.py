from django.urls import path

class ViewContainer:

    def method_view(self, request):
        if False:
            i = 10
            return i + 15
        pass

    @classmethod
    def classmethod_view(cls, request):
        if False:
            while True:
                i = 10
        pass
view_container = ViewContainer()
urlpatterns = [path('', view_container.method_view, name='instance-method-url'), path('', ViewContainer.classmethod_view, name='instance-method-url')]