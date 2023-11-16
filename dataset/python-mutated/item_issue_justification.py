from security_monkey.views import AuthenticatedService
from security_monkey.views import AUDIT_FIELDS
from security_monkey.datastore import ItemAudit
from security_monkey import db, rbac
from flask_restful import marshal
from flask_login import current_user
import datetime

class JustifyPostDelete(AuthenticatedService):
    decorators = [rbac.allow(['Justify'], ['POST', 'DELETE'])]

    def post(self, audit_id):
        if False:
            return 10
        '\n            .. http:post:: /api/1/issues/1234/justification\n\n            Justify an audit issue on a specific item.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                POST /api/1/issues/1234/justification HTTP/1.1\n                Host: example.com\n                Accept: application/json\n\n                {\n                    \'justification\': \'I promise not to abuse this.\'\n                }\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 201 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    "result": {\n                        "justification": "I promise not to abuse this.",\n                        "issue": "Example Issue",\n                        "notes": "Example Notes",\n                        "score": 0,\n                        "item_id": 11890,\n                        "justified_user": "user@example.com",\n                        "justified": true,\n                        "justified_date": "2014-06-19 21:45:58.779168",\n                        "id": 1234\n                    },\n                    "auth": {\n                        "authenticated": true,\n                        "user": "user@example.com"\n                    }\n                }\n\n\n            :statuscode 201: no error\n            :statuscode 401: Authentication Error. Please Login.\n        '
        self.reqparse.add_argument('justification', required=False, type=str, help='Must provide justification', location='json')
        args = self.reqparse.parse_args()
        item = ItemAudit.query.filter(ItemAudit.id == audit_id).first()
        if not item:
            return ({'Error': 'Item with audit_id {} not found'.format(audit_id)}, 404)
        item.justified_user_id = current_user.id
        item.justified = True
        item.justified_date = datetime.datetime.utcnow()
        item.justification = args['justification']
        db.session.add(item)
        db.session.commit()
        db.session.refresh(item)
        retdict = {'auth': self.auth_dict}
        if item.user:
            retdict['result'] = dict(list(marshal(item.__dict__, AUDIT_FIELDS).items()) + list({'justified_user': item.user.email}.items()))
        else:
            retdict['result'] = dict(list(marshal(item.__dict__, AUDIT_FIELDS).items()) + list({'justified_user': None}.items()))
        return (retdict, 200)

    def delete(self, audit_id):
        if False:
            i = 10
            return i + 15
        '\n            .. http:delete:: /api/1/issues/1234/justification\n\n            Remove a justification on an audit issue on a specific item.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                DELETE /api/1/issues/1234/justification HTTP/1.1\n                Host: example.com\n                Accept: application/json\n\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 202 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    "status": "deleted"\n                }\n\n\n            :statuscode 202: Accepted\n            :statuscode 401: Authentication Error. Please Login.\n        '
        item = ItemAudit.query.filter(ItemAudit.id == audit_id).first()
        if not item:
            return ({'Error': 'Item with audit_id {} not found'.format(audit_id)}, 404)
        item.justified_user_id = None
        item.justified = False
        item.justified_date = None
        item.justification = None
        db.session.add(item)
        db.session.commit()
        return ({'status': 'deleted'}, 202)