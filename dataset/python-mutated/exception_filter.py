from django.views.debug import SafeExceptionReporterFilter

class CustomExceptionReporterFilter(SafeExceptionReporterFilter):

    def is_active(self, request):
        if False:
            print('Hello World!')
        return True