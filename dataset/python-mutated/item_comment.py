from six import text_type
from security_monkey.views import AuthenticatedService
from security_monkey.views import ITEM_COMMENT_FIELDS
from security_monkey.datastore import ItemComment
from security_monkey import db, rbac
from flask_restful import marshal
from flask_login import current_user
import datetime

class ItemCommentDelete(AuthenticatedService):
    decorators = [rbac.allow(['Justify'], ['DELETE'])]

    def delete(self, item_id, comment_id):
        if False:
            while True:
                i = 10
        "\n            .. http:delete:: /api/1/items/<int:item_id>/comment/<int:comment_id>\n\n            Deletes an item comment.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                DELETE /api/1/items/1234/comment/7718 HTTP/1.1\n                Host: example.com\n                Accept: application/json\n\n                {\n                }\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 202 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    'status': 'deleted'\n                }\n\n            :statuscode 202: Deleted\n            :statuscode 401: Authentication Error. Please Login.\n        "
        query = ItemComment.query.filter(ItemComment.id == comment_id)
        query.filter(ItemComment.user_id == current_user.id).delete()
        db.session.commit()
        return ({'result': 'success'}, 202)

class ItemCommentGet(AuthenticatedService):
    decorators = [rbac.allow(['View'], ['GET'])]

    def get(self, item_id, comment_id):
        if False:
            return 10
        '\n            .. http:get:: /api/1/items/<int:item_id>/comment/<int:comment_id>\n\n            Retrieves an item comment.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                GET /api/1/items/1234/comment/7718 HTTP/1.1\n                Host: example.com\n                Accept: application/json\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 200 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    \'id\': 7719,\n                    \'date_created\': "2013-10-04 22:01:47",\n                    \'text\': \'This is an Item Comment.\',\n                    \'item_id\': 1111\n                }\n\n            :statuscode 200: Success\n            :statuscode 404: Comment with given ID not found.\n            :statuscode 401: Authentication Error. Please Login.\n        '
        query = ItemComment.query.filter(ItemComment.id == comment_id)
        query = query.filter(ItemComment.item_id == item_id)
        ic = query.first()
        if ic is None:
            return ({'status': 'Item Comment Not Found'}, 404)
        comment_marshaled = marshal(ic.__dict__, ITEM_COMMENT_FIELDS)
        comment_marshaled = dict(list(comment_marshaled.items()) + list({'user': ic.user.email}.items()))
        return (comment_marshaled, 200)

class ItemCommentPost(AuthenticatedService):
    decorators = [rbac.allow(['Comment'], ['POST'])]

    def post(self, item_id):
        if False:
            while True:
                i = 10
        '\n            .. http:post:: /api/1/items/<int:item_id>/comments\n\n            Adds an item comment.\n\n            **Example Request**:\n\n            .. sourcecode:: http\n\n                POST /api/1/items/1234/comments HTTP/1.1\n                Host: example.com\n                Accept: application/json\n\n                {\n                    "text": "This item is my favorite."\n                }\n\n            **Example Response**:\n\n            .. sourcecode:: http\n\n                HTTP/1.1 201 OK\n                Vary: Accept\n                Content-Type: application/json\n\n                {\n                    \'item_id\': 1234,\n                    \'id\': 7718,\n                    \'comment\': \'This item is my favorite.\',\n                    \'user\': \'user@example.com\'\n                }\n                {\n                    "date_created": "2014-10-11 23:03:47.716698",\n                    "id": 1,\n                    "text": "This is an item comment."\n                }\n\n            :statuscode 201: Created\n            :statuscode 401: Authentication Error. Please Login.\n        '
        self.reqparse.add_argument('text', required=False, type=text_type, help='Must provide comment', location='json')
        args = self.reqparse.parse_args()
        ic = ItemComment()
        ic.user_id = current_user.id
        ic.item_id = item_id
        ic.text = args['text']
        ic.date_created = datetime.datetime.utcnow()
        db.session.add(ic)
        db.session.commit()
        db.session.refresh(ic)
        comment_marshaled = marshal(ic.__dict__, ITEM_COMMENT_FIELDS)
        comment_marshaled = dict(list(comment_marshaled.items()) + list({'user': ic.user.email}.items()))
        return (comment_marshaled, 201)