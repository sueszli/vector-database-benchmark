"""
.. module: security_monkey.views.account_pattern_audit_score
    :platform: Unix
    :synopsis: Manages restful view for account pattern audit scores

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>

"""
from six import text_type
from security_monkey.views import AuthenticatedService
from security_monkey.views import ACCOUNT_PATTERN_AUDIT_SCORE_FIELDS
from security_monkey.datastore import AccountPatternAuditScore
from security_monkey.datastore import ItemAuditScore
from security_monkey import db, app, rbac
from flask_restful import marshal, reqparse

class AccountPatternAuditScoreGet(AuthenticatedService):
    decorators = [rbac.allow(['View'], ['GET'])]

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(AccountPatternAuditScoreGet, self).__init__()

    def get(self, auditscores_id):
        if False:
            print('Hello World!')
        '\n            .. http:get:: /api/1/auditscores/<int:auditscores_id>/accountpatternauditscores\n\n            Get a list of override scores for account pattern audit scores.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                GET /api/1/auditscores/123/accountpatternauditscores HTTP/1.1\n                Host: example.com\n                Accept: application/json, text/javascript\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 200 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    count: 1,\n                    items: [\n                        {\n                            "id": 234,\n                            "account_pattern": "AccountPattern",\n                            "score": 8,\n                            itemauditscores_id: 123\n                        },\n                    ],\n                    total: 1,\n                    page: 1,\n                    auth: {\n                        authenticated: true,\n                        user: "user@example.com"\n                    }\n                }\n\n            :statuscode 200: no error\n            :statuscode 401: Authentication failure. Please login.\n        '
        result = ItemAuditScore.query.filter(ItemAuditScore.id == auditscores_id).first()
        if not result:
            return ({'status': 'Override Audit Score with the given ID not found.'}, 404)
        self.reqparse.add_argument('count', type=int, default=30, location='args')
        self.reqparse.add_argument('page', type=int, default=1, location='args')
        args = self.reqparse.parse_args()
        page = args.pop('page', None)
        count = args.pop('count', None)
        result = AccountPatternAuditScore.query.paginate(page, count, error_out=False)
        items = []
        for entry in result.items:
            accountpatternauditscore_marshaled = marshal(entry.__dict__, ACCOUNT_PATTERN_AUDIT_SCORE_FIELDS)
            items.append(accountpatternauditscore_marshaled)
        marshaled_dict = {'total': result.total, 'count': len(items), 'page': result.page, 'items': items, 'auth': self.auth_dict}
        return (marshaled_dict, 200)

class AccountPatternAuditScorePost(AuthenticatedService):
    decorators = [rbac.allow(['Admin'], ['POST'])]

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.reqparse = reqparse.RequestParser()
        super(AccountPatternAuditScorePost, self).__init__()

    def post(self):
        if False:
            while True:
                i = 10
        '\n            .. http:post:: /api/1/accountpatternauditscores\n\n            Create a new override account pattern audit score.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                POST /api/1/accountpatternauditscores HTTP/1.1\n                Host: example.com\n                Accept: application/json\n\n                {\n                    "account_pattern": "AccountPattern",\n                    "score": 8,\n                    "itemauditscores_id": 123\n                }\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 201 Created\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    "id": 234,\n                    "account_pattern": "AccountPattern",\n                    "score": 8,\n                    "itemauditscores_id": 123\n                }\n\n            :statuscode 201: created\n            :statuscode 401: Authentication Error. Please Login.\n        '
        self.reqparse.add_argument('account_type', required=False, type=text_type, location='json')
        self.reqparse.add_argument('account_field', required=True, type=text_type, help='Must provide account field', location='json')
        self.reqparse.add_argument('account_pattern', required=True, type=text_type, help='Must provide account pattern', location='json')
        self.reqparse.add_argument('score', required=True, type=text_type, help='Override score required', location='json')
        self.reqparse.add_argument('itemauditscores_id', required=True, type=text_type, help='Audit Score ID required', location='json')
        args = self.reqparse.parse_args()
        result = ItemAuditScore.query.filter(ItemAuditScore.id == args['itemauditscores_id']).first()
        if not result:
            return ({'status': 'Audit Score ID not found.'}, 404)
        result.add_or_update_pattern_score(args['account_type'], args['account_field'], args['account_pattern'], int(args['score']))
        db.session.commit()
        db.session.refresh(result)
        accountpatternauditscore = result.get_account_pattern_audit_score(args['account_type'], args['account_field'], args['account_pattern'])
        accountpatternauditscore_marshaled = marshal(accountpatternauditscore.__dict__, ACCOUNT_PATTERN_AUDIT_SCORE_FIELDS)
        accountpatternauditscore_marshaled['auth'] = self.auth_dict
        return (accountpatternauditscore_marshaled, 201)

class AccountPatternAuditScoreGetPutDelete(AuthenticatedService):
    decorators = [rbac.allow(['View'], ['GET']), rbac.allow(['Admin'], ['PUT', 'DELETE'])]

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.reqparse = reqparse.RequestParser()
        super(AccountPatternAuditScoreGetPutDelete, self).__init__()

    def get(self, id):
        if False:
            return 10
        '\n            .. http:get:: /api/1/accountpatternauditscores/<int:id>\n\n            Get the overide account pattern audit score with given ID.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                GET /api/1/accountpatternauditscores/234 HTTP/1.1\n                Host: example.com\n                Accept: application/json, text/javascript\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 200 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    "id": 234,\n                    "account_pattern": "AccountPattern",\n                    "score": 8,\n                    "itemauditscores_id": 123\n                    auth: {\n                        authenticated: true,\n                        user: "user@example.com"\n                    }\n                }\n\n            :statuscode 200: no error\n            :statuscode 404: item with given ID not found\n            :statuscode 401: Authentication failure. Please login.\n        '
        app.logger.info('ID: ' + str(id))
        result = AccountPatternAuditScore.query.filter(AccountPatternAuditScore.id == id).first()
        if not result:
            return ({'status': 'Override Account Pattern Audit Score with the given ID not found.'}, 404)
        app.logger.info('RESULT DICT: ' + str(result.__dict__))
        accountpatternauditscore_marshaled = marshal(result.__dict__, ACCOUNT_PATTERN_AUDIT_SCORE_FIELDS)
        accountpatternauditscore_marshaled['auth'] = self.auth_dict
        app.logger.info('RETURN: ' + str(accountpatternauditscore_marshaled))
        return (accountpatternauditscore_marshaled, 200)

    def put(self, id):
        if False:
            i = 10
            return i + 15
        '\n            .. http:put:: /api/1/accountpatternauditscores/<int:id>\n\n            Update override account pattern audit score with the given ID.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                PUT /api/1/accountpatternauditscores/234 HTTP/1.1\n                Host: example.com\n                Accept: application/json, text/javascript\n\n                {\n                    "id": 234,\n                    "account_pattern": "AccountPattern",\n                    "score": 5\n                }\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 200 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    "id": 234,\n                    "account_pattern": "AccountPattern"\n                    "score": 5,\n                    "itemauditscores_id": 123\n                    auth: {\n                        authenticated: true,\n                        user: "user@example.com"\n                    }\n                }\n\n            :statuscode 200: no error\n            :statuscode 404: item with given ID not found\n            :statuscode 401: Authentication failure. Please login.\n        '
        self.reqparse.add_argument('account_type', required=False, type=text_type, help='Must provide account type.', location='json')
        self.reqparse.add_argument('account_field', required=False, type=text_type, help='Must provide account field.', location='json')
        self.reqparse.add_argument('account_pattern', required=False, type=text_type, help='Must provide account pattern.', location='json')
        self.reqparse.add_argument('score', required=False, type=text_type, help='Must provide score.', location='json')
        args = self.reqparse.parse_args()
        result = AccountPatternAuditScore.query.filter(AccountPatternAuditScore.id == id).first()
        if not result:
            return ({'status': 'Override Account Pattern Audit Score with the given ID not found.'}, 404)
        result.account_type = args['account_type']
        result.account_field = args['account_field']
        result.account_pattern = args['account_pattern']
        result.score = int(args['score'])
        db.session.add(result)
        db.session.commit()
        db.session.refresh(result)
        accountpatternauditscore_marshaled = marshal(result.__dict__, ACCOUNT_PATTERN_AUDIT_SCORE_FIELDS)
        accountpatternauditscore_marshaled['auth'] = self.auth_dict
        return (accountpatternauditscore_marshaled, 200)

    def delete(self, id):
        if False:
            return 10
        "\n            .. http:delete:: /api/1/accountpatternauditscores/<int:id>\n\n            Delete an override account pattern audit score\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                DELETE /api/1/accountpatternauditscores/234 HTTP/1.1\n                Host: example.com\n                Accept: application/json\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 202 Accepted\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    'status': 'deleted'\n                }\n\n            :statuscode 202: accepted\n            :statuscode 401: Authentication Error. Please Login.\n        "
        AccountPatternAuditScore.query.filter(AccountPatternAuditScore.id == id).delete()
        db.session.commit()
        return ({'status': 'deleted'}, 202)