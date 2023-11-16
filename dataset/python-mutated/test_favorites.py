from tests import BaseTestCase

class TestQueryFavoriteResource(BaseTestCase):

    def test_favorite(self):
        if False:
            for i in range(10):
                print('nop')
        query = self.factory.create_query()
        rv = self.make_request('post', '/api/queries/{}/favorite'.format(query.id))
        self.assertEqual(rv.status_code, 200)
        rv = self.make_request('get', '/api/queries/{}'.format(query.id))
        self.assertEqual(rv.json['is_favorite'], True)

    def test_duplicate_favorite(self):
        if False:
            return 10
        query = self.factory.create_query()
        rv = self.make_request('post', '/api/queries/{}/favorite'.format(query.id))
        self.assertEqual(rv.status_code, 200)
        rv = self.make_request('post', '/api/queries/{}/favorite'.format(query.id))
        self.assertEqual(rv.status_code, 200)

    def test_unfavorite(self):
        if False:
            i = 10
            return i + 15
        query = self.factory.create_query()
        rv = self.make_request('post', '/api/queries/{}/favorite'.format(query.id))
        rv = self.make_request('delete', '/api/queries/{}/favorite'.format(query.id))
        self.assertEqual(rv.status_code, 200)
        rv = self.make_request('get', '/api/queries/{}'.format(query.id))
        self.assertEqual(rv.json['is_favorite'], False)

class TestQueryFavoriteListResource(BaseTestCase):

    def test_get_favorites(self):
        if False:
            print('Hello World!')
        rv = self.make_request('get', '/api/queries/favorites')
        self.assertEqual(rv.status_code, 200)