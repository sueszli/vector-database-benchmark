from django.test import Client
from django.urls import reverse

def test_healthcheck_endpoint():
    if False:
        print('Hello World!')
    c = Client()
    url = reverse('sentry-api-0-relays-healthcheck')
    response = c.get(url)
    assert response.status_code == 200