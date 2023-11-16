import os
import sys
sys.path.append('/var/task')

def get_django_wsgi(settings_module):
    if False:
        i = 10
        return i + 15
    from django.core.wsgi import get_wsgi_application
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', settings_module)
    import django
    if django.VERSION[0] <= 1 and django.VERSION[1] < 7:
        django.setup()
    return get_wsgi_application()