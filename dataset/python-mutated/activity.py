from __future__ import annotations
import datetime
from typing import Any, Iterable, Optional, Type, TypeVar
from typing_extensions import TypeAlias
from sqlalchemy.orm import relationship, backref
from sqlalchemy import types, Column, ForeignKey, or_, and_, not_, union_all, text
from ckan.common import config
import ckan.model as model
import ckan.model.meta as meta
import ckan.model.domain_object as domain_object
import ckan.model.types as _types
from ckan.model.base import BaseModel
from ckan.lib.dictization import table_dictize
from ckan.types import Context, Query
from ckan.lib.plugins import get_permission_labels
__all__ = ['Activity', 'ActivityDetail']
TActivityDetail = TypeVar('TActivityDetail', bound='ActivityDetail')
QActivity: TypeAlias = 'Query[Activity]'

class Activity(domain_object.DomainObject, BaseModel):
    __tablename__ = 'activity'
    __table_args__ = {'extend_existing': True}
    id = Column('id', types.UnicodeText, primary_key=True, default=_types.make_uuid)
    timestamp = Column('timestamp', types.DateTime)
    user_id = Column('user_id', types.UnicodeText)
    object_id = Column('object_id', types.UnicodeText)
    revision_id = Column('revision_id', types.UnicodeText)
    activity_type = Column('activity_type', types.UnicodeText)
    data = Column('data', _types.JsonDictType)
    permission_labels = Column('permission_labels', types.Text)
    activity_detail: 'ActivityDetail'

    def __init__(self, user_id: str, object_id: str, activity_type: str, data: Optional[dict[str, Any]]=None, permission_labels: Optional[list[str]]=None) -> None:
        if False:
            while True:
                i = 10
        self.id = _types.make_uuid()
        self.timestamp = datetime.datetime.utcnow()
        self.user_id = user_id
        self.object_id = object_id
        self.activity_type = activity_type
        self.permission_labels = permission_labels
        if data is None:
            self.data = {}
        else:
            self.data = data

    @classmethod
    def get(cls, id: str) -> Optional['Activity']:
        if False:
            return 10
        'Returns an Activity object referenced by its id.'
        if not id:
            return None
        return meta.Session.query(cls).get(id)

    @classmethod
    def activity_stream_item(cls, pkg: model.Package, activity_type: str, user_id: str) -> Optional['Activity']:
        if False:
            while True:
                i = 10
        import ckan.model
        import ckan.logic
        assert activity_type in ('new', 'changed'), str(activity_type)
        if activity_type == 'changed' and pkg.state == 'deleted':
            if meta.Session.query(cls).filter_by(object_id=pkg.id, activity_type='deleted').all():
                return None
            else:
                activity_type = 'deleted'
        permission_labels = get_permission_labels().get_dataset_labels(pkg)
        try:
            dictized_package = ckan.logic.get_action('package_show')({'for_view': False, 'ignore_auth': True}, {'id': pkg.id})
        except ckan.logic.NotFound:
            return None
        actor = meta.Session.query(ckan.model.User).get(user_id)
        return cls(user_id, pkg.id, '%s package' % activity_type, {'package': dictized_package, 'actor': actor.name if actor else None}, permission_labels)

def activity_dictize(activity: Activity, context: Context) -> dict[str, Any]:
    if False:
        while True:
            i = 10
    return table_dictize(activity, context)

def activity_list_dictize(activity_list: Iterable[Activity], context: Context) -> list[dict[str, Any]]:
    if False:
        print('Hello World!')
    return [activity_dictize(activity, context) for activity in activity_list]

class ActivityDetail(domain_object.DomainObject):
    __tablename__ = 'activity_detail'
    id = Column('id', types.UnicodeText, primary_key=True, default=_types.make_uuid)
    activity_id = Column('activity_id', types.UnicodeText, ForeignKey('activity.id'))
    object_id = Column('object_id', types.UnicodeText)
    object_type = Column('object_type', types.UnicodeText)
    activity_type = Column('activity_type', types.UnicodeText)
    data = Column('data', _types.JsonDictType)
    activity = relationship(Activity, backref=backref('activity_detail', cascade='all, delete-orphan'))

    def __init__(self, activity_id: str, object_id: str, object_type: str, activity_type: str, data: Optional[dict[str, Any]]=None) -> None:
        if False:
            i = 10
            return i + 15
        self.activity_id = activity_id
        self.object_id = object_id
        self.object_type = object_type
        self.activity_type = activity_type
        if data is None:
            self.data = {}
        else:
            self.data = data

    @classmethod
    def by_activity_id(cls: Type[TActivityDetail], activity_id: str) -> list['TActivityDetail']:
        if False:
            return 10
        return model.Session.query(cls).filter_by(activity_id=activity_id).all()

def _activities_limit(q: QActivity, limit: int, offset: Optional[int]=None, revese_order: Optional[bool]=False) -> QActivity:
    if False:
        while True:
            i = 10
    '\n    Return an SQLAlchemy query for all activities at an offset with a limit.\n\n    revese_order:\n        if we want the last activities before a date, we must reverse the\n        order before limiting.\n    '
    if revese_order:
        q = q.order_by(Activity.timestamp)
    else:
        q = q.order_by(Activity.timestamp.desc())
    if offset:
        q = q.offset(offset)
    if limit:
        q = q.limit(limit)
    return q

def _activities_union_all(*qlist: QActivity) -> QActivity:
    if False:
        for i in range(10):
            print('nop')
    '\n    Return union of two or more activity queries sorted by timestamp,\n    and remove duplicates\n    '
    q: QActivity = model.Session.query(Activity).select_entity_from(union_all(*[q.subquery().select() for q in qlist])).distinct(Activity.timestamp)
    return q

def _activities_from_user_query(user_id: str) -> QActivity:
    if False:
        for i in range(10):
            print('nop')
    'Return an SQLAlchemy query for all activities from user_id.'
    q = model.Session.query(Activity)
    q = q.filter(Activity.user_id == user_id)
    return q

def _activities_about_user_query(user_id: str) -> QActivity:
    if False:
        for i in range(10):
            print('nop')
    'Return an SQLAlchemy query for all activities about user_id.'
    q = model.Session.query(Activity)
    q = q.filter(Activity.object_id == user_id)
    return q

def _user_activity_query(user_id: str, limit: int) -> QActivity:
    if False:
        i = 10
        return i + 15
    'Return an SQLAlchemy query for all activities from or about user_id.'
    q1 = _activities_limit(_activities_from_user_query(user_id), limit)
    q2 = _activities_limit(_activities_about_user_query(user_id), limit)
    return _activities_union_all(q1, q2)

def user_activity_list(user_id: str, limit: int, offset: int, after: Optional[datetime.datetime]=None, before: Optional[datetime.datetime]=None, user_permission_labels: Optional[list[str]]=None) -> list[Activity]:
    if False:
        for i in range(10):
            print('nop')
    'Return user_id\'s public activity stream.\n\n    Return a list of all activities from or about the given user, i.e. where\n    the given user is the subject or object of the activity, e.g.:\n\n    "{USER} created the dataset {DATASET}"\n    "{OTHER_USER} started following {USER}"\n    etc.\n\n    '
    q1 = _activities_from_user_query(user_id)
    q2 = _activities_about_user_query(user_id)
    q = _activities_union_all(q1, q2)
    q = _filter_activitites_from_users(q)
    q = _filter_activities_by_permission_labels(q, user_permission_labels)
    if after:
        q = q.filter(Activity.timestamp > after)
    if before:
        q = q.filter(Activity.timestamp < before)
    revese_order = after and (not before)
    if revese_order:
        q = q.order_by(Activity.timestamp)
    else:
        q = q.order_by(Activity.timestamp.desc())
    if offset:
        q = q.offset(offset)
    if limit:
        q = q.limit(limit)
    results = q.all()
    if revese_order:
        results.reverse()
    return results

def _package_activity_query(package_id: str) -> QActivity:
    if False:
        while True:
            i = 10
    'Return an SQLAlchemy query for all activities about package_id.'
    q = model.Session.query(Activity).filter_by(object_id=package_id)
    return q

def package_activity_list(package_id: str, limit: int, offset: Optional[int]=None, after: Optional[datetime.datetime]=None, before: Optional[datetime.datetime]=None, include_hidden_activity: bool=False, activity_types: Optional[list[str]]=None, exclude_activity_types: Optional[list[str]]=None, user_permission_labels: Optional[list[str]]=None) -> list[Activity]:
    if False:
        print('Hello World!')
    'Return the given dataset (package)\'s public activity stream.\n\n    Returns all activities about the given dataset, i.e. where the given\n    dataset is the object of the activity, e.g.:\n\n    "{USER} created the dataset {DATASET}"\n    "{USER} updated the dataset {DATASET}"\n    etc.\n\n    '
    q = _package_activity_query(package_id)
    if not include_hidden_activity:
        q = _filter_activitites_from_users(q)
    if activity_types:
        q = _filter_activitites_from_type(q, include=True, types=activity_types)
    elif exclude_activity_types:
        q = _filter_activitites_from_type(q, include=False, types=exclude_activity_types)
    q = _filter_activities_by_permission_labels(q, user_permission_labels)
    if after:
        q = q.filter(Activity.timestamp > after)
    if before:
        q = q.filter(Activity.timestamp < before)
    revese_order = after and (not before)
    if revese_order:
        q = q.order_by(Activity.timestamp)
    else:
        q = q.order_by(Activity.timestamp.desc())
    if offset:
        q = q.offset(offset)
    if limit:
        q = q.limit(limit)
    results = q.all()
    if revese_order:
        results.reverse()
    return results

def _group_activity_query(group_id: str) -> QActivity:
    if False:
        while True:
            i = 10
    "Return an SQLAlchemy query for all activities about group_id.\n\n    Returns a query for all activities whose object is either the group itself\n    or one of the group's datasets.\n\n    "
    group = model.Group.get(group_id)
    if not group:
        return model.Session.query(Activity).filter(text('0=1'))
    q: QActivity = model.Session.query(Activity).outerjoin(model.Member, Activity.object_id == model.Member.table_id).outerjoin(model.Package, and_(model.Package.id == model.Member.table_id, model.Package.private == False)).filter(or_(and_(model.Member.group_id == group_id, model.Member.state == 'active', model.Package.state == 'active'), and_(model.Member.group_id == group_id, model.Member.state == 'deleted', model.Package.state == 'deleted'), Activity.object_id == group_id))
    return q

def _organization_activity_query(org_id: str) -> QActivity:
    if False:
        return 10
    "Return an SQLAlchemy query for all activities about org_id.\n\n    Returns a query for all activities whose object is either the org itself\n    or one of the org's datasets.\n\n    "
    org = model.Group.get(org_id)
    if not org or not org.is_organization:
        return model.Session.query(Activity).filter(text('0=1'))
    q: QActivity = model.Session.query(Activity).outerjoin(model.Package, and_(model.Package.id == Activity.object_id, model.Package.private == False)).filter(or_(model.Package.owner_org == org_id, Activity.object_id == org_id))
    return q

def group_activity_list(group_id: str, limit: int, offset: int, after: Optional[datetime.datetime]=None, before: Optional[datetime.datetime]=None, include_hidden_activity: bool=False, activity_types: Optional[list[str]]=None, user_permission_labels: Optional[list[str]]=None) -> list[Activity]:
    if False:
        print('Hello World!')
    'Return the given group\'s public activity stream.\n\n    Returns activities where the given group or one of its datasets is the\n    object of the activity, e.g.:\n\n    "{USER} updated the group {GROUP}"\n    "{USER} updated the dataset {DATASET}"\n    etc.\n\n    '
    q = _group_activity_query(group_id)
    if not include_hidden_activity:
        q = _filter_activitites_from_users(q)
    if activity_types:
        q = _filter_activitites_from_type(q, include=True, types=activity_types)
    q = _filter_activities_by_permission_labels(q, user_permission_labels)
    if after:
        q = q.filter(Activity.timestamp > after)
    if before:
        q = q.filter(Activity.timestamp < before)
    revese_order = after and (not before)
    if revese_order:
        q = q.order_by(Activity.timestamp)
    else:
        q = q.order_by(Activity.timestamp.desc())
    if offset:
        q = q.offset(offset)
    if limit:
        q = q.limit(limit)
    results = q.all()
    if revese_order:
        results.reverse()
    return results

def organization_activity_list(group_id: str, limit: int, offset: int, after: Optional[datetime.datetime]=None, before: Optional[datetime.datetime]=None, include_hidden_activity: bool=False, activity_types: Optional[list[str]]=None, user_permission_labels: Optional[list[str]]=None) -> list[Activity]:
    if False:
        return 10
    'Return the given org\'s public activity stream.\n\n    Returns activities where the given org or one of its datasets is the\n    object of the activity, e.g.:\n\n    "{USER} updated the organization {ORG}"\n    "{USER} updated the dataset {DATASET}"\n    etc.\n\n    '
    q = _organization_activity_query(group_id)
    if not include_hidden_activity:
        q = _filter_activitites_from_users(q)
    if activity_types:
        q = _filter_activitites_from_type(q, include=True, types=activity_types)
    q = _filter_activities_by_permission_labels(q, user_permission_labels)
    if after:
        q = q.filter(Activity.timestamp > after)
    if before:
        q = q.filter(Activity.timestamp < before)
    revese_order = after and (not before)
    if revese_order:
        q = q.order_by(Activity.timestamp)
    else:
        q = q.order_by(Activity.timestamp.desc())
    if offset:
        q = q.offset(offset)
    if limit:
        q = q.limit(limit)
    results = q.all()
    if revese_order:
        results.reverse()
    return results

def _activities_from_users_followed_by_user_query(user_id: str, limit: int) -> QActivity:
    if False:
        print('Hello World!')
    'Return a query for all activities from users that user_id follows.'
    follower_objects = model.UserFollowingUser.followee_list(user_id)
    if not follower_objects:
        return model.Session.query(Activity).filter(text('0=1'))
    return _activities_union_all(*[_user_activity_query(follower.object_id, limit) for follower in follower_objects])

def _activities_from_datasets_followed_by_user_query(user_id: str, limit: int) -> QActivity:
    if False:
        i = 10
        return i + 15
    'Return a query for all activities from datasets that user_id follows.'
    follower_objects = model.UserFollowingDataset.followee_list(user_id)
    if not follower_objects:
        return model.Session.query(Activity).filter(text('0=1'))
    return _activities_union_all(*[_activities_limit(_package_activity_query(follower.object_id), limit) for follower in follower_objects])

def _activities_from_groups_followed_by_user_query(user_id: str, limit: int) -> QActivity:
    if False:
        return 10
    "Return a query for all activities about groups the given user follows.\n\n    Return a query for all activities about the groups the given user follows,\n    or about any of the group's datasets. This is the union of\n    _group_activity_query(group_id) for each of the groups the user follows.\n\n    "
    follower_objects = model.UserFollowingGroup.followee_list(user_id)
    if not follower_objects:
        return model.Session.query(Activity).filter(text('0=1'))
    return _activities_union_all(*[_activities_limit(_group_activity_query(follower.object_id), limit) for follower in follower_objects])

def _activities_from_everything_followed_by_user_query(user_id: str, limit: int=0) -> QActivity:
    if False:
        print('Hello World!')
    'Return a query for all activities from everything user_id follows.'
    q1 = _activities_from_users_followed_by_user_query(user_id, limit)
    q2 = _activities_from_datasets_followed_by_user_query(user_id, limit)
    q3 = _activities_from_groups_followed_by_user_query(user_id, limit)
    return _activities_union_all(q1, q2, q3)

def activities_from_everything_followed_by_user(user_id: str, limit: int, offset: int) -> list[Activity]:
    if False:
        return 10
    'Return activities from everything that the given user is following.\n\n    Returns all activities where the object of the activity is anything\n    (user, dataset, group...) that the given user is following.\n\n    '
    q = _activities_from_everything_followed_by_user_query(user_id, limit + offset)
    return _activities_limit(q, limit, offset).all()

def _dashboard_activity_query(user_id: str, limit: int=0) -> QActivity:
    if False:
        i = 10
        return i + 15
    "Return an SQLAlchemy query for user_id's dashboard activity stream."
    q1 = _user_activity_query(user_id, limit)
    q2 = _activities_from_everything_followed_by_user_query(user_id, limit)
    return _activities_union_all(q1, q2)

def dashboard_activity_list(user_id: str, limit: int, offset: int, before: Optional[datetime.datetime]=None, after: Optional[datetime.datetime]=None, user_permission_labels: Optional[list[str]]=None) -> list[Activity]:
    if False:
        while True:
            i = 10
    "Return the given user's dashboard activity stream.\n\n    Returns activities from the user's public activity stream, plus\n    activities from everything that the user is following.\n\n    This is the union of user_activity_list(user_id) and\n    activities_from_everything_followed_by_user(user_id).\n\n    "
    q = _dashboard_activity_query(user_id)
    q = _filter_activitites_from_users(q)
    q = _filter_activities_by_permission_labels(q, user_permission_labels)
    if after:
        q = q.filter(Activity.timestamp > after)
    if before:
        q = q.filter(Activity.timestamp < before)
    revese_order = after and (not before)
    if revese_order:
        q = q.order_by(Activity.timestamp)
    else:
        q = q.order_by(Activity.timestamp.desc())
    if offset:
        q = q.offset(offset)
    if limit:
        q = q.limit(limit)
    results = q.all()
    if revese_order:
        results.reverse()
    return results

def _changed_packages_activity_query() -> QActivity:
    if False:
        while True:
            i = 10
    "Return an SQLAlchemy query for all changed package activities.\n\n    Return a query for all activities with activity_type '*package', e.g.\n    'new_package', 'changed_package', 'deleted_package'.\n\n    "
    q = model.Session.query(Activity)
    q = q.filter(Activity.activity_type.endswith('package'))
    return q

def recently_changed_packages_activity_list(limit: int, offset: int, user_permission_labels: Optional[list[str]]=None) -> list[Activity]:
    if False:
        print('Hello World!')
    "Return the site-wide stream of recently changed package activities.\n\n    This activity stream includes recent 'new package', 'changed package' and\n    'deleted package' activities for the whole site.\n\n    "
    q = _changed_packages_activity_query()
    q = _filter_activitites_from_users(q)
    q = _filter_activities_by_permission_labels(q, user_permission_labels)
    return _activities_limit(q, limit, offset).all()

def _filter_activitites_from_users(q: QActivity) -> QActivity:
    if False:
        for i in range(10):
            print('nop')
    '\n    Adds a filter to an existing query object to avoid activities from users\n    defined in :ref:`ckan.hide_activity_from_users` (defaults to the site user)\n    '
    users_to_avoid = _activity_stream_get_filtered_users()
    if users_to_avoid:
        q = q.filter(Activity.user_id.notin_(users_to_avoid))
    return q

def _filter_activitites_from_type(q: QActivity, types: Iterable[str], include: bool=True):
    if False:
        i = 10
        return i + 15
    'Adds a filter to an existing query object to include or exclude\n    (include=False) activities based on a list of types.\n\n    '
    if include:
        q = q.filter(Activity.activity_type.in_(types))
    else:
        q = q.filter(Activity.activity_type.notin_(types))
    return q

def _activity_stream_get_filtered_users() -> list[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the list of users from the :ref:`ckan.hide_activity_from_users` config\n    option and return a list of their ids. If the config is not specified,\n    returns the id of the site user.\n    '
    users_list = config.get('ckan.hide_activity_from_users')
    if not users_list:
        from ckan.logic import get_action
        context: Context = {'ignore_auth': True}
        site_user = get_action('get_site_user')(context, {})
        users_list = [site_user.get('name')]
    return model.User.user_ids_for_name_or_id(users_list)

def _filter_activities_by_permission_labels(q: QActivity, user_permission_labels: Optional[list[str]]=None):
    if False:
        while True:
            i = 10
    'Adds a filter to an existing query object to\n    exclude package activities based on user permissions.\n    '
    if user_permission_labels is not None:
        q = q.filter(or_(or_(Activity.activity_type.is_(None), not_(Activity.activity_type.endswith('package'))), Activity.permission_labels.op('&&')(user_permission_labels)))
    return q