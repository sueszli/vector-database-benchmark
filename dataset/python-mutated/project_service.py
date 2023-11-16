from typing import List
from typing import Union
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.linked_obj import LinkedObject
from ...types.uid import UID
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..notification.notification_service import NotificationService
from ..notification.notifications import CreateNotification
from ..response import SyftError
from ..response import SyftNotReady
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from ..user.user_roles import ONLY_DATA_SCIENTIST_ROLE_LEVEL
from ..user.user_roles import ServiceRole
from .project import Project
from .project import ProjectEvent
from .project import ProjectRequest
from .project import ProjectSubmit
from .project import create_project_hash
from .project_stash import ProjectStash

@instrument
@serializable()
class ProjectService(AbstractService):
    store: DocumentStore
    stash: ProjectStash

    def __init__(self, store: DocumentStore) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.store = store
        self.stash = ProjectStash(store=store)

    @service_method(path='project.can_create_project', name='can_create_project', roles=ONLY_DATA_SCIENTIST_ROLE_LEVEL)
    def can_create_project(self, context: AuthedServiceContext) -> Union[bool, SyftError]:
        if False:
            return 10
        user_service = context.node.get_service('userservice')
        role = user_service.get_role_for_credentials(credentials=context.credentials)
        if role == ServiceRole.DATA_SCIENTIST:
            return True
        return SyftError(message='User cannot create projects')

    @service_method(path='project.create_project', name='create_project', roles=ONLY_DATA_SCIENTIST_ROLE_LEVEL)
    def create_project(self, context: AuthedServiceContext, project: ProjectSubmit) -> Union[SyftSuccess, SyftError]:
        if False:
            while True:
                i = 10
        'Start a Project'
        check_role = self.can_create_project(context)
        if isinstance(check_role, SyftError):
            return check_role
        try:
            project_id_check = self.stash.get_by_uid(credentials=context.node.verify_key, uid=project.id)
            if project_id_check.is_err():
                return SyftError(message=f'{project_id_check.err()}')
            if project_id_check.ok() is not None:
                return SyftError(message=f'Project with id: {project.id} already exists.')
            project_obj: Project = project.to(Project, context=context)
            leader_node = project_obj.state_sync_leader
            if leader_node.verify_key != context.node.verify_key:
                network_service = context.node.get_service('networkservice')
                peer = network_service.stash.get_for_verify_key(credentials=context.node.verify_key, verify_key=leader_node.verify_key)
                if peer.is_err():
                    return SyftError(message=f'Leader Node(id={leader_node.id.short()}) is not a peer of this Node(id={context.node.id.short()})')
                leader_node_peer = peer.ok()
            else:
                leader_node_peer = project.leader_node_route.validate_with_context(context=context)
            project_obj.leader_node_peer = leader_node_peer
            project_obj.start_hash = create_project_hash(project_obj)[1]
            result = self.stash.set(context.credentials, project_obj)
            if result.is_err():
                return SyftError(message=str(result.err()))
            project_obj_store = result.ok()
            project_obj_store = self.add_signing_key_to_project(context, project_obj_store)
            return project_obj_store
        except Exception as e:
            print('Failed to submit Project', e)
            raise e

    @service_method(path='project.add_event', name='add_event', roles=GUEST_ROLE_LEVEL)
    def add_event(self, context: AuthedServiceContext, project_event: ProjectEvent) -> Union[SyftSuccess, SyftError]:
        if False:
            while True:
                i = 10
        'To add events to a projects'
        project_obj = self.stash.get_by_uid(context.node.verify_key, uid=project_event.project_id)
        if project_obj.is_ok():
            project: Project = project_obj.ok()
            if project.state_sync_leader.verify_key == context.node.verify_key:
                return SyftError(message='Project Events should be passed to leader by broadcast endpoint')
            if context.credentials != project.state_sync_leader.verify_key:
                return SyftError(message='Only the leader of the project can add events')
            project.events.append(project_event)
            project.event_id_hashmap[project_event.id] = project_event
            message_result = self.check_for_project_request(project, project_event, context)
            if isinstance(message_result, SyftError):
                return message_result
            result = self.stash.update(context.node.verify_key, project)
            if result.is_err():
                return SyftError(message=str(result.err()))
            return SyftSuccess(message=f'Project event {project_event.id} added successfully ')
        if project_obj.is_err():
            return SyftError(message=str(project_obj.err()))

    @service_method(path='project.broadcast_event', name='broadcast_event', roles=GUEST_ROLE_LEVEL)
    def broadcast_event(self, context: AuthedServiceContext, project_event: ProjectEvent) -> Union[SyftSuccess, SyftError]:
        if False:
            i = 10
            return i + 15
        'To add events to a projects'
        project_obj = self.stash.get_by_uid(context.node.verify_key, uid=project_event.project_id)
        if project_obj.is_err():
            return SyftError(message=str(project_obj.err()))
        project = project_obj.ok()
        if not project.has_permission(context.credentials):
            return SyftError(message='User does not have permission to add events')
        project: Project = project_obj.ok()
        if project.state_sync_leader.verify_key != context.node.verify_key:
            return SyftError(message='Only the leader of the project can broadcast events')
        if project_event.seq_no <= len(project.events) and len(project.events) > 0:
            return SyftNotReady(message='Project out of sync event')
        if project_event.seq_no > len(project.events) + 1:
            return SyftError(message='Project event out of order!')
        project.events.append(project_event)
        project.event_id_hashmap[project_event.id] = project_event
        message_result = self.check_for_project_request(project, project_event, context)
        if isinstance(message_result, SyftError):
            return message_result
        network_service = context.node.get_service('networkservice')
        for member in project.members:
            if member.verify_key != context.node.verify_key:
                peer = network_service.stash.get_for_verify_key(credentials=context.node.verify_key, verify_key=member.verify_key)
                if peer.is_err():
                    return SyftError(message=f'Leader node does not have peer {member.name}-{member.id.short()}' + ' Kindly exchange routes with the peer')
                peer = peer.ok()
                client = peer.client_with_context(context)
                event_result = client.api.services.project.add_event(project_event)
                if isinstance(event_result, SyftError):
                    return event_result
        result = self.stash.update(context.node.verify_key, project)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message='Successfully Broadcasted Event')

    @service_method(path='project.sync', name='sync', roles=GUEST_ROLE_LEVEL)
    def sync(self, context: AuthedServiceContext, project_id: UID, seq_no: int) -> Union[SyftSuccess, SyftError, List[ProjectEvent]]:
        if False:
            print('Hello World!')
        'To fetch unsynced events from the project'
        project_obj = self.stash.get_by_uid(context.node.verify_key, uid=project_id)
        if project_obj.is_ok():
            project: Project = project_obj.ok()
            if project.state_sync_leader.verify_key != context.node.verify_key:
                return SyftError(message='Project Events should be synced only with the leader')
            if not project.has_permission(context.credentials):
                return SyftError(message='User does not have permission to sync events')
            if seq_no < 0:
                return SyftError(message='Input seq_no should be a non negative integer')
            return project.events[seq_no:]
        if project_obj.is_err():
            return SyftError(message=str(project_obj.err()))

    @service_method(path='project.get_all', name='get_all', roles=GUEST_ROLE_LEVEL)
    def get_all(self, context: AuthedServiceContext) -> Union[List[Project], SyftError]:
        if False:
            print('Hello World!')
        result = self.stash.get_all(context.credentials)
        if result.is_err():
            return SyftError(message=str(result.err()))
        projects = result.ok()
        for (idx, project) in enumerate(projects):
            result = self.add_signing_key_to_project(context, project)
            if isinstance(result, SyftError):
                return result
            projects[idx] = result
        return projects

    @service_method(path='project.get_by_name', name='get_by_name', roles=GUEST_ROLE_LEVEL)
    def get_by_name(self, context: AuthedServiceContext, name: str) -> Union[Project, SyftError]:
        if False:
            while True:
                i = 10
        result = self.stash.get_by_name(context.credentials, project_name=name)
        if result.is_err():
            return SyftError(message=str(result.err()))
        elif result.ok():
            project = result.ok()
            return self.add_signing_key_to_project(context, project)
        return SyftError(message=f'Project(name="{name}") does not exist')

    @service_method(path='project.get_by_uid', name='get_by_uid', roles=GUEST_ROLE_LEVEL)
    def get_by_uid(self, context: AuthedServiceContext, uid: UID):
        if False:
            for i in range(10):
                print('nop')
        result = self.stash.get_by_uid(credentials=context.node.verify_key, uid=uid)
        if result.is_err():
            return SyftError(message=str(result.err()))
        elif result.ok():
            return result.ok()
        return SyftError(message=f'Project(id="{uid}") does not exist')

    def add_signing_key_to_project(self, context: AuthedServiceContext, project: Project) -> Union[Project, SyftError]:
        if False:
            print('Hello World!')
        user_service = context.node.get_service('userservice')
        user = user_service.stash.get_by_verify_key(credentials=context.credentials, verify_key=context.credentials)
        if user.is_err():
            return SyftError(message=str(user.err()))
        user = user.ok()
        if not user:
            return SyftError(message='User not found! Kindly register user first')
        project.user_signing_key = user.signing_key
        return project

    def check_for_project_request(self, project: Project, project_event: ProjectEvent, context: AuthedServiceContext):
        if False:
            return 10
        'To check for project request event and create a message for the root user\n\n        Args:\n            project (Project): Project object\n            project_event (ProjectEvent): Project event object\n            context (AuthedServiceContext): Context of the node\n\n        Returns:\n            Union[SyftSuccess, SyftError]: SyftSuccess if message is created else SyftError\n        '
        if isinstance(project_event, ProjectRequest) and project_event.linked_request.node_uid == context.node.id:
            link = LinkedObject.with_context(project, context=context)
            message = CreateNotification(subject=f'A new request has been added to the Project: {project.name}.', from_user_verify_key=context.credentials, to_user_verify_key=context.node.verify_key, linked_obj=link)
            method = context.node.get_service_method(NotificationService.send)
            result = method(context=context, notification=message)
            if isinstance(result, SyftError):
                return result
        return SyftSuccess(message='Successfully Validated Project Request')
TYPE_TO_SERVICE[Project] = ProjectService
SERVICE_TO_TYPES[ProjectService].update({Project})