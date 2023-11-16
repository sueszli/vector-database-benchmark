from typing import List
from typing import Union
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectREAD
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from .notification_stash import NotificationStash
from .notifications import CreateNotification
from .notifications import LinkedObject
from .notifications import Notification
from .notifications import NotificationStatus
from .notifications import ReplyNotification

@instrument
@serializable()
class NotificationService(AbstractService):
    store: DocumentStore
    stash: NotificationStash

    def __init__(self, store: DocumentStore) -> None:
        if False:
            while True:
                i = 10
        self.store = store
        self.stash = NotificationStash(store=store)

    @service_method(path='notifications.send', name='send')
    def send(self, context: AuthedServiceContext, notification: CreateNotification) -> Union[Notification, SyftError]:
        if False:
            i = 10
            return i + 15
        'Send a new notification'
        new_notification = notification.to(Notification, context=context)
        permissions = [ActionObjectREAD(uid=new_notification.id, credentials=new_notification.to_user_verify_key)]
        result = self.stash.set(context.credentials, new_notification, add_permissions=permissions)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()

    @service_method(path='notifications.reply', name='reply', roles=GUEST_ROLE_LEVEL)
    def reply(self, context: AuthedServiceContext, reply: ReplyNotification) -> Union[ReplyNotification, SyftError]:
        if False:
            for i in range(10):
                print('nop')
        msg = self.stash.get_by_uid(credentials=context.credentials, uid=reply.target_msg)
        if msg.is_ok():
            msg = msg.ok()
            reply.from_user_verify_key = context.credentials
            msg.replies.append(reply)
            result = self.stash.update(credentials=context.credentials, obj=msg)
            if result.is_ok():
                return result.ok()
            else:
                SyftError(message="Couldn't add a new notification reply in the target notification.")
        else:
            SyftError(message='The target notification id {reply.target_msg} was not found!')

    @service_method(path='notifications.get_all', name='get_all', roles=DATA_SCIENTIST_ROLE_LEVEL)
    def get_all(self, context: AuthedServiceContext) -> Union[List[Notification], SyftError]:
        if False:
            for i in range(10):
                print('nop')
        result = self.stash.get_all_inbox_for_verify_key(context.credentials, verify_key=context.credentials)
        if result.err():
            return SyftError(message=str(result.err()))
        notifications = result.ok()
        return notifications

    @service_method(path='notifications.get_all_sent', name='outbox', roles=DATA_SCIENTIST_ROLE_LEVEL)
    def get_all_sent(self, context: AuthedServiceContext) -> Union[List[Notification], SyftError]:
        if False:
            print('Hello World!')
        result = self.stash.get_all_sent_for_verify_key(context.credentials, context.credentials)
        if result.err():
            return SyftError(message=str(result.err()))
        notifications = result.ok()
        return notifications

    def get_all_for_status(self, context: AuthedServiceContext, status: NotificationStatus) -> Union[List[Notification], SyftError]:
        if False:
            while True:
                i = 10
        result = self.stash.get_all_by_verify_key_for_status(context.credentials, verify_key=context.credentials, status=status)
        if result.err():
            return SyftError(message=str(result.err()))
        notifications = result.ok()
        return notifications

    @service_method(path='notifications.get_all_read', name='get_all_read', roles=DATA_SCIENTIST_ROLE_LEVEL)
    def get_all_read(self, context: AuthedServiceContext) -> Union[List[Notification], SyftError]:
        if False:
            return 10
        return self.get_all_for_status(context=context, status=NotificationStatus.READ)

    @service_method(path='notifications.get_all_unread', name='get_all_unread', roles=DATA_SCIENTIST_ROLE_LEVEL)
    def get_all_unread(self, context: AuthedServiceContext) -> Union[List[Notification], SyftError]:
        if False:
            return 10
        return self.get_all_for_status(context=context, status=NotificationStatus.UNREAD)

    @service_method(path='notifications.mark_as_read', name='mark_as_read')
    def mark_as_read(self, context: AuthedServiceContext, uid: UID) -> Union[Notification, SyftError]:
        if False:
            while True:
                i = 10
        result = self.stash.update_notification_status(context.credentials, uid=uid, status=NotificationStatus.READ)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()

    @service_method(path='notifications.mark_as_unread', name='mark_as_unread')
    def mark_as_unread(self, context: AuthedServiceContext, uid: UID) -> Union[Notification, SyftError]:
        if False:
            while True:
                i = 10
        result = self.stash.update_notification_status(context.credentials, uid=uid, status=NotificationStatus.UNREAD)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()

    @service_method(path='notifications.resolve_object', name='resolve_object', roles=DATA_SCIENTIST_ROLE_LEVEL)
    def resolve_object(self, context: AuthedServiceContext, linked_obj: LinkedObject) -> Union[Notification, SyftError]:
        if False:
            while True:
                i = 10
        service = context.node.get_service(linked_obj.service_type)
        result = service.resolve_link(context=context, linked_obj=linked_obj)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()

    @service_method(path='notifications.clear', name='clear')
    def clear(self, context: AuthedServiceContext) -> Union[SyftError, SyftSuccess]:
        if False:
            while True:
                i = 10
        result = self.stash.delete_all_for_verify_key(credentials=context.credentials, verify_key=context.credentials)
        if result.is_ok():
            return SyftSuccess(message='All notifications cleared !!')
        return SyftError(message=str(result.err()))

    def filter_by_obj(self, context: AuthedServiceContext, obj_uid: UID) -> Union[Notification, SyftError]:
        if False:
            print('Hello World!')
        notifications = self.stash.get_all(context.credentials)
        if notifications.is_ok():
            for notification in notifications.ok():
                if notification.linked_obj and notification.linked_obj.object_uid == obj_uid:
                    return notification
        else:
            return SyftError(message='Could not get notifications!!')
TYPE_TO_SERVICE[Notification] = NotificationService
SERVICE_TO_TYPES[NotificationService].update({Notification})