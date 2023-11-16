import unittest
from django.db import connection, connections, transaction
from django.http import Http404
from django.test import TestCase, TransactionTestCase, override_settings
from django.urls import path
from rest_framework import status
from rest_framework.exceptions import APIException
from rest_framework.response import Response
from rest_framework.test import APIRequestFactory
from rest_framework.views import APIView
from tests.models import BasicModel
factory = APIRequestFactory()

class BasicView(APIView):

    def post(self, request, *args, **kwargs):
        if False:
            while True:
                i = 10
        BasicModel.objects.create()
        return Response({'method': 'GET'})

class ErrorView(APIView):

    def post(self, request, *args, **kwargs):
        if False:
            return 10
        BasicModel.objects.create()
        raise Exception

class APIExceptionView(APIView):

    def post(self, request, *args, **kwargs):
        if False:
            return 10
        BasicModel.objects.create()
        raise APIException

class NonAtomicAPIExceptionView(APIView):

    @transaction.non_atomic_requests
    def dispatch(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return super().dispatch(*args, **kwargs)

    def get(self, request, *args, **kwargs):
        if False:
            return 10
        BasicModel.objects.all()
        raise Http404
urlpatterns = (path('', NonAtomicAPIExceptionView.as_view()),)

@unittest.skipUnless(connection.features.uses_savepoints, "'atomic' requires transactions and savepoints.")
class DBTransactionTests(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.view = BasicView.as_view()
        connections.databases['default']['ATOMIC_REQUESTS'] = True

    def tearDown(self):
        if False:
            return 10
        connections.databases['default']['ATOMIC_REQUESTS'] = False

    def test_no_exception_commit_transaction(self):
        if False:
            for i in range(10):
                print('nop')
        request = factory.post('/')
        with self.assertNumQueries(1):
            response = self.view(request)
        assert not transaction.get_rollback()
        assert response.status_code == status.HTTP_200_OK
        assert BasicModel.objects.count() == 1

@unittest.skipUnless(connection.features.uses_savepoints, "'atomic' requires transactions and savepoints.")
class DBTransactionErrorTests(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.view = ErrorView.as_view()
        connections.databases['default']['ATOMIC_REQUESTS'] = True

    def tearDown(self):
        if False:
            while True:
                i = 10
        connections.databases['default']['ATOMIC_REQUESTS'] = False

    def test_generic_exception_delegate_transaction_management(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Transaction is eventually managed by outer-most transaction atomic\n        block. DRF do not try to interfere here.\n\n        We let django deal with the transaction when it will catch the Exception.\n        '
        request = factory.post('/')
        with self.assertNumQueries(3):
            with transaction.atomic():
                self.assertRaises(Exception, self.view, request)
                assert not transaction.get_rollback()
        assert BasicModel.objects.count() == 1

@unittest.skipUnless(connection.features.uses_savepoints, "'atomic' requires transactions and savepoints.")
class DBTransactionAPIExceptionTests(TestCase):

    def setUp(self):
        if False:
            return 10
        self.view = APIExceptionView.as_view()
        connections.databases['default']['ATOMIC_REQUESTS'] = True

    def tearDown(self):
        if False:
            return 10
        connections.databases['default']['ATOMIC_REQUESTS'] = False

    def test_api_exception_rollback_transaction(self):
        if False:
            return 10
        '\n        Transaction is rollbacked by our transaction atomic block.\n        '
        request = factory.post('/')
        num_queries = 4 if connection.features.can_release_savepoints else 3
        with self.assertNumQueries(num_queries):
            with transaction.atomic():
                response = self.view(request)
                assert transaction.get_rollback()
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert BasicModel.objects.count() == 0

@unittest.skipUnless(connection.features.uses_savepoints, "'atomic' requires transactions and savepoints.")
class MultiDBTransactionAPIExceptionTests(TestCase):
    databases = '__all__'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.view = APIExceptionView.as_view()
        connections.databases['default']['ATOMIC_REQUESTS'] = True
        connections.databases['secondary']['ATOMIC_REQUESTS'] = True

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        connections.databases['default']['ATOMIC_REQUESTS'] = False
        connections.databases['secondary']['ATOMIC_REQUESTS'] = False

    def test_api_exception_rollback_transaction(self):
        if False:
            i = 10
            return i + 15
        '\n        Transaction is rollbacked by our transaction atomic block.\n        '
        request = factory.post('/')
        num_queries = 4 if connection.features.can_release_savepoints else 3
        with self.assertNumQueries(num_queries):
            with transaction.atomic(), transaction.atomic(using='secondary'):
                response = self.view(request)
                assert transaction.get_rollback()
                assert transaction.get_rollback(using='secondary')
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert BasicModel.objects.count() == 0

@unittest.skipUnless(connection.features.uses_savepoints, "'atomic' requires transactions and savepoints.")
@override_settings(ROOT_URLCONF='tests.test_atomic_requests')
class NonAtomicDBTransactionAPIExceptionTests(TransactionTestCase):

    def setUp(self):
        if False:
            return 10
        connections.databases['default']['ATOMIC_REQUESTS'] = True

    def tearDown(self):
        if False:
            return 10
        connections.databases['default']['ATOMIC_REQUESTS'] = False

    def test_api_exception_rollback_transaction_non_atomic_view(self):
        if False:
            return 10
        response = self.client.get('/')
        assert response.status_code == status.HTTP_404_NOT_FOUND