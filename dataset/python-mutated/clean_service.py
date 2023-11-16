from datetime import datetime, timedelta
from django.utils.timezone import make_aware
from django_drf_filepond.models import TemporaryUpload

def clean_service():
    if False:
        for i in range(10):
            print('nop')
    tus = TemporaryUpload.objects.filter(uploaded__lte=make_aware(datetime.now() - timedelta(days=1)))
    tus.delete()