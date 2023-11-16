from typing import Any
import uuid
import six
import datetime
from sqlalchemy.sql import select
from sqlalchemy import Table, Column, ForeignKey, Boolean, UnicodeText, Text, String, DateTime, and_, inspect
import sqlalchemy.orm.properties
from sqlalchemy.orm import class_mapper
from sqlalchemy.orm import relation
import ckan.logic as logic
import ckan.lib.dictization as d
from ckan.lib.dictization.model_dictize import _execute, resource_list_dictize, extras_list_dictize, group_list_dictize
from ckan import model
from ckanext.activity.model import Activity

def package_dictize_with_revisions(pkg, context, include_plugin_data=False):
    if False:
        return 10
    "\n    Given a Package object, returns an equivalent dictionary.\n\n    Normally this is the most recent version, but you can provide revision_id\n    or revision_date in the context and it will filter to an earlier time.\n\n    May raise NotFound if:\n    * the specified revision_id doesn't exist\n    * the specified revision_date was before the package was created\n    "
    model = context['model']
    try:
        model.PackageRevision
        revision_model = model
    except AttributeError:
        revision_model = RevisionTableMappings.instance()
    is_latest_revision = not (context.get(u'revision_id') or context.get(u'revision_date'))
    execute = _execute if is_latest_revision else _execute_with_revision
    if is_latest_revision:
        if isinstance(pkg, revision_model.PackageRevision):
            pkg = model.Package.get(pkg.id)
        result = pkg
    else:
        package_rev = revision_model.package_revision_table
        q = select([package_rev]).where(package_rev.c.id == pkg.id)
        result = execute(q, package_rev, context).first()
    if not result:
        raise logic.NotFound
    result_dict = d.table_dictize(result, context)
    if result_dict.get(u'title'):
        result_dict['title'] = result_dict['title'].strip()
    if is_latest_revision:
        res = model.resource_table
    else:
        res = revision_model.resource_revision_table
    mm_col = res._columns.get(u'metadata_modified')
    if mm_col is not None:
        res._columns.remove(mm_col)
    q = select([res]).where(res.c.package_id == pkg.id)
    result = execute(q, res, context)
    result_dict['resources'] = resource_list_dictize(result, context)
    result_dict['num_resources'] = len(result_dict.get(u'resources', []))
    tag = model.tag_table
    if is_latest_revision:
        pkg_tag = model.package_tag_table
    else:
        pkg_tag = revision_model.package_tag_revision_table
    q = select([tag, pkg_tag.c.state], from_obj=pkg_tag.join(tag, tag.c.id == pkg_tag.c.tag_id)).where(pkg_tag.c.package_id == pkg.id)
    result = execute(q, pkg_tag, context)
    result_dict['tags'] = d.obj_list_dictize(result, context, lambda x: x['name'])
    result_dict['num_tags'] = len(result_dict.get(u'tags', []))
    for tag in result_dict['tags']:
        assert u'display_name' not in tag
        tag['display_name'] = tag['name']
    if is_latest_revision:
        extra = model.package_extra_table
    else:
        extra = revision_model.extra_revision_table
    q = select([extra]).where(extra.c.package_id == pkg.id)
    result = execute(q, extra, context)
    result_dict['extras'] = extras_list_dictize(result, context)
    if is_latest_revision:
        member = model.member_table
    else:
        member = revision_model.member_revision_table
    group = model.group_table
    q = select([group, member.c.capacity], from_obj=member.join(group, group.c.id == member.c.group_id)).where(member.c.table_id == pkg.id).where(member.c.state == u'active').where(group.c.is_organization == False)
    result = execute(q, member, context)
    context['with_capacity'] = False
    result_dict['groups'] = group_list_dictize(result, context, with_package_counts=False)
    if is_latest_revision:
        group = model.group_table
    else:
        group = revision_model.group_revision_table
    q = select([group]).where(group.c.id == result_dict['owner_org']).where(group.c.state == u'active')
    result = execute(q, group, context)
    organizations = d.obj_list_dictize(result, context)
    if organizations:
        result_dict['organization'] = organizations[0]
    else:
        result_dict['organization'] = None
    if is_latest_revision:
        rel = model.package_relationship_table
    else:
        rel = revision_model.package_relationship_revision_table
    q = select([rel]).where(rel.c.subject_package_id == pkg.id)
    result = execute(q, rel, context)
    result_dict['relationships_as_subject'] = d.obj_list_dictize(result, context)
    q = select([rel]).where(rel.c.object_package_id == pkg.id)
    result = execute(q, rel, context)
    result_dict['relationships_as_object'] = d.obj_list_dictize(result, context)
    result_dict['isopen'] = pkg.isopen if isinstance(pkg.isopen, bool) else pkg.isopen()
    result_dict['type'] = pkg.type or u'dataset'
    if pkg.license and pkg.license.url:
        result_dict['license_url'] = pkg.license.url
        result_dict['license_title'] = pkg.license.title.split(u'::')[-1]
    elif pkg.license:
        result_dict['license_title'] = pkg.license.title
    else:
        result_dict['license_title'] = pkg.license_id
    if is_latest_revision:
        result_dict['metadata_modified'] = pkg.metadata_modified.isoformat()
    result_dict['metadata_created'] = pkg.metadata_created.isoformat()
    return result_dict

def _execute_with_revision(q, rev_table, context):
    if False:
        i = 10
        return i + 15
    "\n    Takes an SqlAlchemy query (q) that is (at its base) a Select on an object\n    revision table (rev_table), and you provide revision_id or revision_date in\n    the context and it will filter the object revision(s) to an earlier time.\n\n    Raises NotFound if context['revision_id'] is provided, but the revision\n    ID does not exist.\n\n    Returns [] if there are no results.\n\n    "
    model = context['model']
    session = model.Session
    revision_id = context.get(u'revision_id')
    revision_date = context.get(u'revision_date')
    if revision_id:
        revision = session.query(revision_model.Revision).filter_by(id=revision_id).first()
        if not revision:
            raise logic.NotFound
        revision_date = revision.timestamp
    q = q.where(rev_table.c.revision_timestamp <= revision_date)
    q = q.where(rev_table.c.expired_timestamp > revision_date)
    return session.execute(q)

def make_revisioned_table(base_table, frozen=False):
    if False:
        for i in range(10):
            print('nop')
    "Modify base_table and create correponding revision table.\n\n    A 'frozen' revision table is not written to any more - it's just there\n    as a record. It doesn't have the continuity foreign key relation.\n\n    @return revision table (e.g. package_revision)\n    "
    revision_table = Table(base_table.name + u'_revision', base_table.metadata)
    copy_table(base_table, revision_table)
    revision_table.append_column(Column(u'revision_id', UnicodeText, ForeignKey(u'revision.id')))
    pkcols = []
    for col in base_table.c:
        if col.primary_key:
            pkcols.append(col)
    assert len(pkcols) <= 1, u'Do not support versioning objects with multiple primary keys'
    fk_name = base_table.name + u'.' + pkcols[0].name
    revision_table.append_column(Column(u'continuity_id', pkcols[0].type, None if frozen else ForeignKey(fk_name)))
    for col in revision_table.c:
        if col.name == u'revision_id':
            col.primary_key = True
            revision_table.primary_key.columns.add(col)
    revision_table.append_column(Column(u'expired_id', Text))
    revision_table.append_column(Column(u'revision_timestamp', DateTime))
    revision_table.append_column(Column(u'expired_timestamp', DateTime, default=datetime.datetime(9999, 12, 31)))
    revision_table.append_column(Column(u'current', Boolean))
    return revision_table

def copy_column(name, src_table, dest_table):
    if False:
        i = 10
        return i + 15
    col = src_table.c[name]
    if col.unique is True:
        col.unique = False
    dest_table.append_column(col.copy())
    newcol = dest_table.c[name]
    if len(col.foreign_keys) > 0:
        for fk in col.foreign_keys:
            newcol.append_foreign_key(fk.copy())

def copy_table_columns(table):
    if False:
        print('Hello World!')
    columns = []
    for col in table.c:
        newcol = col.copy()
        if len(col.foreign_keys) > 0:
            for fk in col.foreign_keys:
                newcol.foreign_keys.add(fk.copy())
        columns.append(newcol)
    return columns

def copy_table(table, newtable):
    if False:
        while True:
            i = 10
    for key in table.c.keys():
        if key != 'plugin_data':
            copy_column(key, table, newtable)

def make_revision_table(metadata):
    if False:
        while True:
            i = 10
    revision_table = Table(u'revision', metadata, Column(u'id', UnicodeText, primary_key=True, default=lambda : six.u(uuid.uuid4())), Column(u'timestamp', DateTime, default=datetime.datetime.utcnow), Column(u'author', String(200)), Column(u'message', UnicodeText), Column(u'state', UnicodeText, default=model.State.ACTIVE))
    return revision_table

def make_Revision(mapper, revision_table):
    if False:
        return 10
    mapper(Revision, revision_table, properties={})
    return Revision

class Revision(object):
    """A Revision to the Database/Domain Model.

    All versioned objects have an associated Revision which can be accessed via
    the revision attribute.
    """

    def __init__(self, **kw):
        if False:
            print('Hello World!')
        for (k, v) in kw.items():
            setattr(self, k, v)

def create_object_version(mapper_fn, base_object, rev_table):
    if False:
        return 10
    'Create the Version Domain Object corresponding to base_object.\n\n    E.g. if Package is our original object we should do::\n\n        # name of Version Domain Object class\n        PackageVersion = create_object_version(..., Package, ...)\n\n    NB: This must obviously be called after mapping has happened to\n    base_object.\n    '

    class MyClass(object):

        def __init__(self, **kw):
            if False:
                for i in range(10):
                    print('nop')
            for (k, v) in kw.items():
                setattr(self, k, v)
    name = base_object.__name__ + u'Revision'
    MyClass.__name__ = str(name)
    MyClass.__continuity_class__ = base_object
    base_object.__revision_class__ = MyClass
    ourmapper = mapper_fn(MyClass, rev_table)
    base_mapper = class_mapper(base_object)
    for prop in base_mapper.iterate_properties:
        try:
            is_relation = prop.__class__ == sqlalchemy.orm.properties.PropertyLoader
        except AttributeError:
            is_relation = prop.__class__ == sqlalchemy.orm.properties.RelationshipProperty
        if is_relation:
            prop_remote_obj = prop.argument
            remote_obj_is_revisioned = getattr(prop_remote_obj, u'__revisioned__', False)
            is_many = prop.secondary is not None or prop.uselist
            if remote_obj_is_revisioned:
                propname = prop.key
                add_fake_relation(MyClass, propname, is_many=is_many)
            elif not is_many:
                ourmapper.add_property(prop.key, relation(prop_remote_obj))
            else:
                pass
    return MyClass

def add_fake_relation(revision_class, name, is_many=False):
    if False:
        print('Hello World!')
    "Add a 'fake' relation on ObjectRevision objects.\n\n    These relation are fake in that they just proxy to the continuity object\n    relation.\n    "

    def _pget(self):
        if False:
            for i in range(10):
                print('nop')
        related_object = getattr(self.continuity, name)
        if is_many:
            return related_object
        else:
            return related_object.get_as_of()
    x = property(_pget)
    setattr(revision_class, name, x)

def make_package_revision(package):
    if False:
        print('Hello World!')
    'Manually create a revision for a package and its related objects\n    '
    instances = [package]
    package_tags = model.Session.query(model.PackageTag).filter_by(package_id=package.id).all()
    instances.extend(package_tags)
    extras = model.Session.query(model.PackageExtra).filter_by(package_id=package.id).all()
    instances.extend(extras)
    instances.extend(package.resources)
    instances.extend(package.get_groups())
    members = model.Session.query(model.Member).filter_by(table_id=package.id).all()
    instances.extend(members)
    make_revision(instances)

def make_revision(instances):
    if False:
        for i in range(10):
            print('nop')
    'Manually create a revision.\n\n    Copies a new/changed row from a table (e.g. Package) into its\n    corresponding revision table (e.g. PackageRevision) and makes an entry\n    in the Revision table.\n    '
    Revision = RevisionTableMappings.instance().Revision
    revision = Revision()
    model.Session.add(revision)
    revision.id = str(uuid.uuid4())
    model.Session.add(revision)
    model.Session.flush()
    for instance in instances:
        colvalues = {}
        mapper = inspect(type(instance))
        table = mapper.tables[0]
        for key in table.c.keys():
            val = getattr(instance, key)
            colvalues[key] = val
        colvalues['revision_id'] = revision.id
        colvalues['continuity_id'] = instance.id
        revision_table = RevisionTableMappings.instance().revision_table_mapping[type(instance)]
        ins = revision_table.insert().values(colvalues)
        model.Session.execute(ins)
    activity = model.Session.query(Activity).order_by(Activity.timestamp.desc()).first()
    activity.revision_id = revision.id
    model.Session.flush()
    for instance in instances:
        if not hasattr(instance, u'__revision_class__'):
            continue
        revision_cls = instance.__revision_class__
        revision_table = RevisionTableMappings.instance().revision_table_mapping[type(instance)]
        model.Session.execute(revision_table.update().where(and_(revision_table.c.id == instance.id, revision_table.c.current is True)).values(current=False))
        q = model.Session.query(revision_cls)
        q = q.filter_by(expired_timestamp=datetime.datetime(9999, 12, 31), id=instance.id)
        results = q.all()
        for rev_obj in results:
            values = {}
            if rev_obj.revision_id == revision.id:
                values['revision_timestamp'] = revision.timestamp
            else:
                values['expired_timestamp'] = revision.timestamp
            model.Session.execute(revision_table.update().where(and_(revision_table.c.id == rev_obj.id, revision_table.c.revision_id == rev_obj.revision_id)).values(**values))

class RevisionTableMappings(object):
    _instance: Any = None

    @classmethod
    def instance(cls):
        if False:
            i = 10
            return i + 15
        if not cls._instance:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if False:
            while True:
                i = 10
        self.revision_table = make_revision_table(model.meta.metadata)
        self.revision_table.append_column(Column(u'approved_timestamp', DateTime))
        self.Revision = make_Revision(model.meta.mapper, self.revision_table)
        self.package_revision_table = make_revisioned_table(model.package_table)
        self.PackageRevision = create_object_version(model.meta.mapper, model.Package, self.package_revision_table)
        self.resource_revision_table = make_revisioned_table(model.resource_table)
        self.ResourceRevision = create_object_version(model.meta.mapper, model.Resource, self.resource_revision_table)
        self.extra_revision_table = make_revisioned_table(model.package_extra_table)
        self.PackageExtraRevision = create_object_version(model.meta.mapper, model.PackageExtra, self.extra_revision_table)
        self.package_tag_revision_table = make_revisioned_table(model.package_tag_table)
        self.PackageTagRevision = create_object_version(model.meta.mapper, model.PackageTag, self.package_tag_revision_table)
        self.member_revision_table = make_revisioned_table(model.member_table)
        self.MemberRevision = create_object_version(model.meta.mapper, model.Member, self.member_revision_table)
        self.group_revision_table = make_revisioned_table(model.group_table)
        self.GroupRevision = create_object_version(model.meta.mapper, model.Group, self.group_revision_table)
        self.group_extra_revision_table = make_revisioned_table(model.group_extra_table)
        self.GroupExtraRevision = create_object_version(model.meta.mapper, model.GroupExtra, self.group_extra_revision_table)
        self.package_relationship_revision_table = make_revisioned_table(model.package_relationship_table)
        self.system_info_revision_table = make_revisioned_table(model.system_info_table)
        self.SystemInfoRevision = create_object_version(model.meta.mapper, model.SystemInfo, self.system_info_revision_table)
        self.revision_table_mapping = {model.Package: self.package_revision_table, model.Resource: self.resource_revision_table, model.PackageExtra: self.extra_revision_table, model.PackageTag: self.package_tag_revision_table, model.Member: self.member_revision_table, model.Group: self.group_revision_table}
try:
    model.PackageExtraRevision
    revision_model = model
except AttributeError:
    revision_model = RevisionTableMappings.instance()