from six import text_type
from security_monkey.views import AuthenticatedService
from security_monkey.views import REVISION_COMMENT_FIELDS
from security_monkey.datastore import ItemRevisionComment
from security_monkey import db, rbac
from flask_restful import marshal
from flask_login import current_user
import datetime

class RevisionCommentGet(AuthenticatedService):
    decorators = [rbac.allow(['Comment'], ['GET'])]

    def __init__(self):
        if False:
            print('Hello World!')
        super(RevisionCommentGet, self).__init__()

    def get(self, revision_id, comment_id):
        if False:
            while True:
                i = 10
        '\n            .. http:get:: /api/1/revisions/<int:revision_id>/comments/<int:comment_id>\n\n            Get a specific Revision Comment\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                GET /api/1/revisions/1141/comments/22 HTTP/1.1\n                Host: example.com\n                Accept: application/json\n\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 200 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    \'id\': 22,\n                    \'revision_id\': 1141,\n                    "date_created": "2013-10-04 22:01:47",\n                    \'text\': \'This is a Revision Comment.\'\n                }\n\n            :statuscode 200: no error\n            :statuscode 404: Revision Comment with given ID not found.\n            :statuscode 401: Authentication Error. Please Login.\n        '
        query = ItemRevisionComment.query.filter(ItemRevisionComment.id == comment_id)
        query = query.filter(ItemRevisionComment.revision_id == revision_id)
        irc = query.first()
        if irc is None:
            return ({'status': 'Revision Comment Not Found'}, 404)
        revision_marshaled = marshal(irc.__dict__, REVISION_COMMENT_FIELDS)
        revision_marshaled = dict(list(revision_marshaled.items()) + list({'user': irc.user.email}.items()))
        return (revision_marshaled, 200)

class RevisionCommentDelete(AuthenticatedService):
    decorators = [rbac.allow(['Comment'], ['DELETE'])]

    def __init__(self):
        if False:
            print('Hello World!')
        super(RevisionCommentDelete, self).__init__()

    def delete(self, revision_id, comment_id):
        if False:
            i = 10
            return i + 15
        '\n            .. http:delete:: /api/1/revisions/<int:revision_id>/comments/<int:comment_id>\n\n            Delete a specific Revision Comment\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                DELETE /api/1/revisions/1141/comments/22 HTTP/1.1\n                Host: example.com\n                Accept: application/json\n\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 200 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    \'status\': "deleted"\n                }\n\n            :statuscode 202: Comment Deleted\n            :statuscode 404: Revision Comment with given ID not found.\n            :statuscode 401: Authentication Error. Please Login.\n        '
        query = ItemRevisionComment.query.filter(ItemRevisionComment.id == comment_id)
        query = query.filter(ItemRevisionComment.revision_id == revision_id)
        irc = query.first()
        if irc is None:
            return ({'status': 'Revision Comment Not Found'}, 404)
        query.delete()
        db.session.commit()
        return ({'status': 'deleted'}, 202)

class RevisionCommentPost(AuthenticatedService):
    decorators = [rbac.allow(['Comment'], ['POST'])]

    def post(self, revision_id):
        if False:
            i = 10
            return i + 15
        '\n            .. http:post:: /api/1/revisions/<int:revision_id>/comments\n\n            Create a new Revision Comment.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                POST /api/1/revisions/1141/comments HTTP/1.1\n                Host: example.com\n                Accept: application/json\n\n                {\n                    "text": "This is a Revision Comment."\n                }\n\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 201 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    \'id\': 22,\n                    \'revision_id\': 1141,\n                    "date_created": "2013-10-04 22:01:47",\n                    \'text\': \'This is a Revision Comment.\'\n                }\n\n            :statuscode 201: Revision Comment Created\n            :statuscode 401: Authentication Error. Please Login.\n        '
        self.reqparse.add_argument('text', required=False, type=text_type, help='Must provide comment', location='json')
        args = self.reqparse.parse_args()
        irc = ItemRevisionComment()
        irc.user_id = current_user.id
        irc.revision_id = revision_id
        irc.text = args['text']
        irc.date_created = datetime.datetime.utcnow()
        db.session.add(irc)
        db.session.commit()
        db.session.refresh(irc)
        revision_marshaled = marshal(irc.__dict__, REVISION_COMMENT_FIELDS)
        revision_marshaled = dict(list(revision_marshaled.items()) + list({'user': irc.user.email}.items()))
        return (revision_marshaled, 200)