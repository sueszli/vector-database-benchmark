from six import text_type
from security_monkey.views import AuthenticatedService
from security_monkey.views import IGNORELIST_FIELDS
from security_monkey.datastore import IgnoreListEntry
from security_monkey.datastore import Technology
from security_monkey import db, rbac
from flask_restful import marshal, reqparse

class IgnoreListGetPutDelete(AuthenticatedService):
    decorators = [rbac.allow(['Admin'], ['GET', 'PUT', 'DELETE']), rbac.allow(['View'], ['GET'])]

    def __init__(self):
        if False:
            return 10
        self.reqparse = reqparse.RequestParser()
        super(IgnoreListGetPutDelete, self).__init__()

    def get(self, item_id):
        if False:
            return 10
        '\n            .. http:get:: /api/1/ignorelistentries/<int:id>\n\n            Get the ignorelist entry with the given ID.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                GET /api/1/ignorelistentries/123 HTTP/1.1\n                Host: example.com\n                Accept: application/json, text/javascript\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 200 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    "id": 123,\n                    "prefix": "noisy_",\n                    "notes": "Security Monkey shouldn\'t track noisy_* objects",\n                    "technology": "securitygroup",\n                    auth: {\n                        authenticated: true,\n                        user: "user@example.com"\n                    }\n                }\n\n            :statuscode 200: no error\n            :statuscode 404: item with given ID not found\n            :statuscode 401: Authentication failure. Please login.\n        '
        result = IgnoreListEntry.query.filter(IgnoreListEntry.id == item_id).first()
        if not result:
            return ({'status': 'Ignorelist entry with the given ID not found.'}, 404)
        ignorelistentry_marshaled = marshal(result.__dict__, IGNORELIST_FIELDS)
        ignorelistentry_marshaled['technology'] = result.technology.name
        ignorelistentry_marshaled['auth'] = self.auth_dict
        return (ignorelistentry_marshaled, 200)

    def put(self, item_id):
        if False:
            for i in range(10):
                print('nop')
        '\n            .. http:get:: /api/1/ignorelistentries/<int:id>\n\n            Update the ignorelist entry with the given ID.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                PUT /api/1/ignorelistentries/123 HTTP/1.1\n                Host: example.com\n                Accept: application/json, text/javascript\n\n                {\n                    "id": 123,\n                    "prefix": "noisy_",\n                    "notes": "Security Monkey shouldn\'t track noisy_* objects",\n                    "technology": "securitygroup"\n                }\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 200 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    "id": 123,\n                    "prefix": "noisy_",\n                    "notes": "Security Monkey shouldn\'t track noisy_* objects",\n                    "technology": "securitygroup",\n                    auth: {\n                        authenticated: true,\n                        user: "user@example.com"\n                    }\n                }\n\n            :statuscode 200: no error\n            :statuscode 404: item with given ID not found\n            :statuscode 401: Authentication failure. Please login.\n        '
        self.reqparse.add_argument('prefix', required=True, type=text_type, help='A prefix must be provided which matches the objects you wish to ignore.', location='json')
        self.reqparse.add_argument('notes', required=False, type=text_type, help='Add context.', location='json')
        self.reqparse.add_argument('technology', required=True, type=text_type, help='Technology name required.', location='json')
        args = self.reqparse.parse_args()
        prefix = args['prefix']
        technology = args.get('technology', True)
        notes = args.get('notes', None)
        result = IgnoreListEntry.query.filter(IgnoreListEntry.id == item_id).first()
        if not result:
            return ({'status': 'Ignore list entry with the given ID not found.'}, 404)
        result.prefix = prefix
        result.notes = notes
        technology = Technology.query.filter(Technology.name == technology).first()
        if not technology:
            return ({'status': 'Could not find a technology with the given name'}, 500)
        result.tech_id = technology.id
        db.session.add(result)
        db.session.commit()
        db.session.refresh(result)
        ignorelistentry_marshaled = marshal(result.__dict__, IGNORELIST_FIELDS)
        ignorelistentry_marshaled['technology'] = result.technology.name
        ignorelistentry_marshaled['auth'] = self.auth_dict
        return (ignorelistentry_marshaled, 200)

    def delete(self, item_id):
        if False:
            return 10
        "\n            .. http:delete:: /api/1/ignorelistentries/123\n\n            Delete a ignorelist entry.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                DELETE /api/1/ignorelistentries/123 HTTP/1.1\n                Host: example.com\n                Accept: application/json\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 202 Accepted\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    'status': 'deleted'\n                }\n\n            :statuscode 202: accepted\n            :statuscode 401: Authentication Error. Please Login.\n        "
        IgnoreListEntry.query.filter(IgnoreListEntry.id == item_id).delete()
        db.session.commit()
        return ({'status': 'deleted'}, 202)

class IgnorelistListPost(AuthenticatedService):
    decorators = [rbac.allow(['Admin'], ['GET', 'POST']), rbac.allow(['View'], ['GET'])]

    def get(self):
        if False:
            while True:
                i = 10
        '\n            .. http:get:: /api/1/ignorelistentries\n\n            Get a list of Ignorelist entries.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                GET /api/1/ignorelistentries HTTP/1.1\n                Host: example.com\n                Accept: application/json, text/javascript\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 200 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    count: 1,\n                    items: [\n                        {\n                            "id": 123,\n                            "prefix": "noisy_",\n                            "notes": "Security Monkey shouldn\'t track noisy_* objects",\n                            "technology": "securitygroup"\n                        },\n                    ],\n                    total: 1,\n                    page: 1,\n                    auth: {\n                        authenticated: true,\n                        user: "user@example.com"\n                    }\n                }\n\n            :statuscode 200: no error\n            :statuscode 401: Authentication failure. Please login.\n        '
        self.reqparse.add_argument('count', type=int, default=30, location='args')
        self.reqparse.add_argument('page', type=int, default=1, location='args')
        args = self.reqparse.parse_args()
        page = args.pop('page', None)
        count = args.pop('count', None)
        result = IgnoreListEntry.query.order_by(IgnoreListEntry.id).paginate(page, count, error_out=False)
        items = []
        for entry in result.items:
            ignorelistentry_marshaled = marshal(entry.__dict__, IGNORELIST_FIELDS)
            ignorelistentry_marshaled['technology'] = entry.technology.name
            items.append(ignorelistentry_marshaled)
        marshaled_dict = {'total': result.total, 'count': len(items), 'page': result.page, 'items': items, 'auth': self.auth_dict}
        return (marshaled_dict, 200)

    def post(self):
        if False:
            for i in range(10):
                print('nop')
        '\n            .. http:post:: /api/1/ignorelistentries\n\n            Create a new ignore list entry.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                POST /api/1/ignorelistentries HTTP/1.1\n                Host: example.com\n                Accept: application/json\n\n                {\n                    "prefix": "noisy_",\n                    "notes": "Security Monkey shouldn\'t track noisy_* objects",\n                    "technology": "securitygroup"\n                }\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 201 Created\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    "id": 123,\n                    "prefix": "noisy_",\n                    "notes": "Security Monkey shouldn\'t track noisy_* objects",\n                    "technology": "securitygroup"\n                }\n\n            :statuscode 201: created\n            :statuscode 401: Authentication Error. Please Login.\n        '
        self.reqparse.add_argument('prefix', required=True, type=text_type, help='A prefix must be provided which matches the objects you wish to ignore.', location='json')
        self.reqparse.add_argument('notes', required=False, type=text_type, help='Add context.', location='json')
        self.reqparse.add_argument('technology', required=True, type=text_type, help='Technology name required.', location='json')
        args = self.reqparse.parse_args()
        prefix = args['prefix']
        technology = args.get('technology', True)
        notes = args.get('notes', None)
        entry = IgnoreListEntry()
        entry.prefix = prefix
        if notes:
            entry.notes = notes
        technology = Technology.query.filter(Technology.name == technology).first()
        if not technology:
            return ({'status': 'Could not find a technology with the given name'}, 500)
        entry.tech_id = technology.id
        db.session.add(entry)
        db.session.commit()
        db.session.refresh(entry)
        ignorelistentry_marshaled = marshal(entry.__dict__, IGNORELIST_FIELDS)
        ignorelistentry_marshaled['technology'] = entry.technology.name
        ignorelistentry_marshaled['auth'] = self.auth_dict
        return (ignorelistentry_marshaled, 201)