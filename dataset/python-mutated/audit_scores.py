"""
.. module: security_monkey.views.audit_scores
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from six import text_type
from security_monkey.views import AuthenticatedService
from security_monkey.views import AUDIT_SCORE_FIELDS
from security_monkey.views import ACCOUNT_PATTERN_AUDIT_SCORE_FIELDS
from security_monkey.datastore import ItemAuditScore
from security_monkey import db, rbac
from flask_restful import marshal, reqparse

class AuditScoresGet(AuthenticatedService):
    decorators = [rbac.allow(['View'], ['GET']), rbac.allow(['Admin'], ['POST'])]

    def __init__(self):
        if False:
            return 10
        super(AuditScoresGet, self).__init__()

    def get(self):
        if False:
            return 10
        '\n            .. http:get:: /api/1/auditscores\n\n            Get a list of override scores for audit items.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                GET /api/1/auditscores HTTP/1.1\n                Host: example.com\n                Accept: application/json, text/javascript\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 200 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    count: 1,\n                    items: [\n                        {\n                            "id": 123,\n                            "method": "check_xxx",\n                            "technology": "policy",\n                            "score": 1\n                        },\n                    ],\n                    total: 1,\n                    page: 1,\n                    auth: {\n                        authenticated: true,\n                        user: "user@example.com"\n                    }\n                }\n\n            :statuscode 200: no error\n            :statuscode 401: Authentication failure. Please login.\n        '
        self.reqparse.add_argument('count', type=int, default=30, location='args')
        self.reqparse.add_argument('page', type=int, default=1, location='args')
        args = self.reqparse.parse_args()
        page = args.pop('page', None)
        count = args.pop('count', None)
        result = ItemAuditScore.query.order_by(ItemAuditScore.technology).paginate(page, count, error_out=False)
        items = []
        for entry in result.items:
            auditscore_marshaled = marshal(entry.__dict__, AUDIT_SCORE_FIELDS)
            items.append(auditscore_marshaled)
        marshaled_dict = {'total': result.total, 'count': len(items), 'page': result.page, 'items': items, 'auth': self.auth_dict}
        return (marshaled_dict, 200)

    def post(self):
        if False:
            while True:
                i = 10
        '\n            .. http:post:: /api/1/auditscores\n\n            Create a new override audit score.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                POST /api/1/auditscores HTTP/1.1\n                Host: example.com\n                Accept: application/json\n\n                {\n                    "method": "check_xxx",\n                    "technology": "policy",\n                    "score": 1\n                }\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 201 Created\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    "id": 123,\n                    "name": "Corp",\n                    "notes": "Corporate Network",\n                    "cidr": "1.2.3.4/22"\n                }\n\n            :statuscode 201: created\n            :statuscode 401: Authentication Error. Please Login.\n        '
        self.reqparse.add_argument('method', required=True, type=text_type, help='Must provide method name', location='json')
        self.reqparse.add_argument('technology', required=True, type=text_type, help='Technology required.', location='json')
        self.reqparse.add_argument('score', required=False, type=text_type, help='Override score required', location='json')
        self.reqparse.add_argument('disabled', required=True, type=text_type, help='Disabled flag', location='json')
        args = self.reqparse.parse_args()
        method = args['method']
        technology = args['technology']
        score = args['score']
        if score is None:
            score = 0
        disabled = args['disabled']
        query = ItemAuditScore.query.filter(ItemAuditScore.technology == technology)
        query = query.filter(ItemAuditScore.method == method)
        auditscore = query.first()
        if not auditscore:
            auditscore = ItemAuditScore()
            auditscore.method = method
            auditscore.technology = technology
        auditscore.score = int(score)
        auditscore.disabled = bool(disabled)
        db.session.add(auditscore)
        db.session.commit()
        db.session.refresh(auditscore)
        auditscore_marshaled = marshal(auditscore.__dict__, AUDIT_SCORE_FIELDS)
        auditscore_marshaled['auth'] = self.auth_dict
        return (auditscore_marshaled, 201)

class AuditScoreGetPutDelete(AuthenticatedService):
    decorators = [rbac.allow(['View'], ['GET']), rbac.allow(['Admin'], ['PUT', 'DELETE'])]

    def __init__(self):
        if False:
            return 10
        self.reqparse = reqparse.RequestParser()
        super(AuditScoreGetPutDelete, self).__init__()

    def get(self, id):
        if False:
            print('Hello World!')
        '\n            .. http:get:: /api/1/auditscores/<int:id>\n\n            Get the overide audit score with given ID.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                GET /api/1/auditscores/123 HTTP/1.1\n                Host: example.com\n                Accept: application/json, text/javascript\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 200 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    "id": 123,\n                    "method": "check_xxx",\n                    "technology": "policy",\n                    "score": "1",\n                    auth: {\n                        authenticated: true,\n                        user: "user@example.com"\n                    }\n                }\n\n            :statuscode 200: no error\n            :statuscode 404: item with given ID not found\n            :statuscode 401: Authentication failure. Please login.\n        '
        result = ItemAuditScore.query.filter(ItemAuditScore.id == id).first()
        if not result:
            return ({'status': 'Override Audit Score with the given ID not found.'}, 404)
        auditscore_marshaled = marshal(result.__dict__, AUDIT_SCORE_FIELDS)
        auditscore_marshaled['auth'] = self.auth_dict
        account_pattern_scores_marshaled = []
        for account_pattern_score in result.account_pattern_scores:
            account_pattern_score_marshaled = marshal(account_pattern_score, ACCOUNT_PATTERN_AUDIT_SCORE_FIELDS)
            account_pattern_scores_marshaled.append(account_pattern_score_marshaled)
        auditscore_marshaled['account_pattern_scores'] = account_pattern_scores_marshaled
        return (auditscore_marshaled, 200)

    def put(self, id):
        if False:
            return 10
        '\n            .. http:get:: /api/1/auditscores/<int:id>\n\n            Update override audit score with the given ID.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                PUT /api/1/auditscores/123 HTTP/1.1\n                Host: example.com\n                Accept: application/json, text/javascript\n\n                {\n                    "id": 123,\n                    "method": "check_xxx",\n                    "technology": "policy",\n                    "Score": "1"\n                }\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 200 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    "id": 123,\n                    "score": "1",\n                    auth: {\n                        authenticated: true,\n                        user: "user@example.com"\n                    }\n                }\n\n            :statuscode 200: no error\n            :statuscode 404: item with given ID not found\n            :statuscode 401: Authentication failure. Please login.\n        '
        self.reqparse.add_argument('method', required=True, type=text_type, help='Must provide method name', location='json')
        self.reqparse.add_argument('technology', required=True, type=text_type, help='Technology required.', location='json')
        self.reqparse.add_argument('score', required=False, type=text_type, help='Must provide score.', location='json')
        self.reqparse.add_argument('disabled', required=True, type=text_type, help='Must disabled flag.', location='json')
        args = self.reqparse.parse_args()
        score = args['score']
        if score is None:
            score = 0
        result = ItemAuditScore.query.filter(ItemAuditScore.id == id).first()
        if not result:
            return ({'status': 'Override audit score with the given ID not found.'}, 404)
        result.method = args['method']
        result.technology = args['technology']
        result.disabled = args['disabled']
        result.score = int(score)
        db.session.add(result)
        db.session.commit()
        db.session.refresh(result)
        auditscore_marshaled = marshal(result.__dict__, AUDIT_SCORE_FIELDS)
        auditscore_marshaled['auth'] = self.auth_dict
        return (auditscore_marshaled, 200)

    def delete(self, id):
        if False:
            while True:
                i = 10
        "\n            .. http:delete:: /api/1/auditscores/123\n\n            Delete an override audit score\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                DELETE /api/1/auditscores/123 HTTP/1.1\n                Host: example.com\n                Accept: application/json\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 202 Accepted\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    'status': 'deleted'\n                }\n\n            :statuscode 202: accepted\n            :statuscode 401: Authentication Error. Please Login.\n        "
        result = ItemAuditScore.query.filter(ItemAuditScore.id == id).first()
        db.session.delete(result)
        db.session.commit()
        return ({'status': 'deleted'}, 202)