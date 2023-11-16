"""API endpoints related to notifications."""
import sqlalchemy
from flask import request
from flask_restx import Namespace, Resource, marshal
from sqlalchemy.orm import joinedload, noload
from app import models, schema, utils
from app.connections import db
from app.core import notifications
from app.core.notifications import webhooks
api = Namespace('notifications', description='Orchest-api notifications API.')
api = schema.register_schema(api)
logger = utils.get_logger()

@api.route('/subscribable-events')
class SubscribableEventList(Resource):

    @api.doc('subscribable-events')
    def get(self):
        if False:
            print('Hello World!')
        'Gets all subscribable events.'
        return ({'events': notifications.get_subscribable_events()}, 200)

@api.route('/subscribers')
class SubscriberList(Resource):

    @api.doc('get_subscribers')
    @api.response(200, 'Success', schema.subscribers)
    def get(self):
        if False:
            return 10
        "Gets all subscribers, doesn't include their subscriptions."
        subscribers = models.Subscriber.query.options(noload(models.Subscriber.subscriptions)).filter(models.Subscriber.type != 'analytics').all()
        marshaled = []
        for subscriber in subscribers:
            if isinstance(subscriber, models.Webhook):
                marshaled.append(marshal(subscriber, schema.webhook))
            else:
                marshaled.append(marshal(subscriber, schema.subscriber))
        return ({'subscribers': marshaled}, 200)

@api.route('/subscribers/webhooks')
class WebhookList(Resource):

    @api.doc('create_webhook')
    @api.expect(schema.webhook_spec, validate=True)
    @api.response(201, 'Success', schema.webhook)
    def post(self):
        if False:
            for i in range(10):
                print('nop')
        "Creates a webhook with the given subscriptions.\n\n        Repeated subscription entries are ignored. If no secret is\n        passed a secret will be generated through the BE. This endpoint\n        returns a model without the secret. All other endpoints also do\n        not return the secret, meaning that it's not possible to get\n        back a secret from the BE, for security reasons.\n        "
        try:
            webhook = webhooks.create_webhook(request.get_json())
        except (ValueError, sqlalchemy.exc.IntegrityError) as e:
            return ({'message': str(e)}, 400)
        db.session.commit()
        return (marshal(webhook, schema.webhook), 201)

@api.route('/subscribers/webhooks/<string:uuid>')
class Webhook(Resource):

    @api.doc('update_webhook')
    @api.expect(schema.webhook_mutation, validate=True)
    @api.response(200, 'Success', schema.webhook)
    def put(self, uuid: str):
        if False:
            for i in range(10):
                print('nop')
        'Updates a webhook, including its subscriptions.\n\n        The mutation only contains the values of a webhook to be\n        changed. The original value will remain unchanged if not\n        mentioned in the mutation.\n        '
        try:
            webhooks.update_webhook(uuid, request.get_json())
        except ValueError as e:
            return ({'message': f'Invalid payload. {e}'}, 400)
        except Exception as e:
            return ({'message': f'Failed to update webhook. {e}'}, 500)
        db.session.commit()
        return ({'message': f'Webhook {uuid} has been updated.'}, 200)

@api.route('/subscribers/webhooks/pre-creation-test-ping-delivery')
class WebhookPreCreationTestPingDelivery(Resource):

    @api.doc('pre_creation_test_ping')
    @api.expect(schema.webhook_spec, validate=True)
    @api.response(200, 'Success')
    @api.response(500, 'Failure')
    def post(self):
        if False:
            i = 10
            return i + 15
        'Send a test ping delivery to a webhook before creating it.\n\n        This endpoint allows to send a test ping delivery to a given\n        webhook spec, to allow testing delivery before creating the\n        webhook.\n\n        The endpoint will return a 200 if the response obtained from the\n        deliveree is to be considered successful, 500 otherwise.\n\n        '
        try:
            webhook_spec = request.get_json()
            webhook_spec['subscriptions'] = []
            webhook = webhooks.create_webhook(webhook_spec)
            response = webhooks.send_test_ping_delivery(webhook.uuid)
            if response is not None and response.status_code >= 200 and (response.status_code <= 299):
                return ({'message': 'success'}, 200)
            else:
                if response is not None:
                    logger.info(response.status_code)
                    logger.info(response.text)
                return ({'message': 'failure'}, 500)
        except (ValueError, sqlalchemy.exc.IntegrityError) as e:
            return ({'message': str(e)}, 400)
        finally:
            db.session.rollback()

@api.route('/subscribers/<string:uuid>')
class Subscriber(Resource):

    @api.doc('subscriber')
    @api.response(200, 'Success', schema.subscriber)
    @api.response(200, 'Success', schema.webhook)
    def get(self, uuid: str):
        if False:
            i = 10
            return i + 15
        'Gets a subscriber, including its subscriptions.'
        subscriber = models.Subscriber.query.options(joinedload(models.Subscriber.subscriptions)).filter(models.Subscriber.uuid == uuid).first()
        if subscriber is None:
            return ({'message': f'Subscriber {uuid} does not exist.'}, 404)
        if isinstance(subscriber, models.Webhook):
            subscriber = marshal(subscriber, schema.webhook)
        else:
            subscriber = marshal(subscriber, schema.subscriber)
        return (subscriber, 200)

    @api.doc('delete_subscriber')
    def delete(self, uuid: str):
        if False:
            return 10
        models.Subscriber.query.filter(models.Subscriber.uuid == uuid).delete()
        db.session.commit()
        return ({'message': ''}, 201)

@api.route('/subscribers/test-ping-delivery/<string:uuid>')
class SendSubscriberTestPingDelivery(Resource):

    @api.doc('subscribers/test-ping-delivery')
    @api.response(200, 'Success')
    @api.response(500, 'Failure')
    def get(self, uuid: str):
        if False:
            while True:
                i = 10
        "Send a test ping delivery to the subscriber.\n\n        This endpoint allows to send a ping event notifications to the\n        subscriber, so that it's possible to test if a given webhook\n        is working end to end, i.e. the deliveree is reachable.\n\n        The endpoint will return a 200 if the response obtained from the\n        deliveree is to be considered successfull, 500 otherwise.\n\n        "
        response = webhooks.send_test_ping_delivery(uuid)
        if response is not None and response.status_code >= 200 and (response.status_code <= 299):
            return ({'message': 'success'}, 200)
        else:
            if response is not None:
                logger.info(response.status_code)
                logger.info(response.text)
            return ({'message': 'failure'}, 500)

@api.route('/subscribers/subscribed-to/<string:event_type>')
class SubscribersSubscribedToEvent(Resource):

    @api.doc('get_subscribers_subscribed_to_event')
    @api.response(200, 'Success', schema.subscribers)
    @api.doc('get_subscribers_subscribed_to_event', params={'project_uuid': {'description': 'Optional, uuid of the project to which the event is related.', 'type': str}, 'job_uuid': {'description': "Optional, uuid of the job to which the event is related, if provided, 'project_uuid' must be provided as well.", 'type': str}})
    def get(self, event_type: str):
        if False:
            return 10
        'Gets all subscribers subscribed to a given event_type.\n\n        Not passing anything (i.e. just specifying a `event_type`\n        through the path, no project/job uuid) means that you will be\n        querying for subscribers that are subscribed to the event\n        "globally", i.e. not specific to a project or a job, which\n        means that subscribers subscribed to a given event for a\n        specific project or job would not come up in the result. If you\n        know which project or job you are querying for you should\n        specify it.\n\n        This can be useful to know if, for example, a given job failure\n        would lead to any notification whatsoever.\n\n        Args:\n            event_type: An event_type from the list at\n                `/subscribable-events`, note that it must be\n                percent-encoded, see\n                https://developer.mozilla.org/en-US/docs/Glossary/percent-encoding.\n        '
        try:
            alerted_subscribers = notifications.get_subscribers_subscribed_to_event(event_type, request.args.get('project_uuid'), request.args.get('job_uuid'))
        except ValueError as e:
            return ({'message': str(e)}, 400)
        marshaled = []
        for subscriber in alerted_subscribers:
            if isinstance(subscriber, models.Webhook):
                marshaled.append(marshal(subscriber, schema.webhook))
            elif isinstance(subscriber, models.AnalyticsSubscriber):
                continue
            else:
                marshaled.append(marshal(subscriber, schema.subscriber))
        return ({'subscribers': marshaled}, 200)