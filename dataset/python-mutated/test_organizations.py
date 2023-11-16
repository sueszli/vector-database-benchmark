import pretend
from warehouse import db
from warehouse.cli import organizations
from warehouse.events.tags import EventTag
from warehouse.organizations.models import Organization, OrganizationApplication, OrganizationNameCatalog
from ...common.db.accounts import UserEventFactory
from ...common.db.organizations import OrganizationEventFactory, OrganizationFactory, OrganizationRoleFactory

def raise_(ex):
    if False:
        print('Hello World!')
    '\n    Used by lambda functions to raise exception\n    '
    raise ex

def test_migrate(db_request, monkeypatch, cli):
    if False:
        for i in range(10):
            print('nop')
    engine = pretend.stub()
    config = pretend.stub(registry={'sqlalchemy.engine': engine})
    session_cls = pretend.call_recorder(lambda bind: db_request.db)
    monkeypatch.setattr(db, 'Session', session_cls)
    db_request.db.commit = db_request.db.flush
    approved = OrganizationFactory(is_approved=True)
    declined = OrganizationFactory(is_approved=False)
    submitted = OrganizationFactory(is_approved=None)
    for org in [approved, declined, submitted]:
        role = OrganizationRoleFactory.create(organization=org)
        UserEventFactory.create(source=role.user, tag=EventTag.Account.OrganizationRoleAdd, additional={'organization_name': org.name, 'role_name': 'Owner'})
        OrganizationEventFactory.create(source=org, tag=EventTag.Organization.OrganizationRoleAdd)
        OrganizationEventFactory.create(source=org, tag=EventTag.Organization.OrganizationCreate, additional={'created_by_user_id': str(role.user.id)})
    result = cli.invoke(organizations.migrate_unapproved_orgs_to_applications, obj=config)
    assert result.exit_code == 0
    assert db_request.db.query(OrganizationNameCatalog).count() == 1
    assert db_request.db.query(Organization).count() == 1
    assert db_request.db.query(OrganizationApplication).filter_by(is_approved=True).count() == 1
    assert db_request.db.query(OrganizationApplication).filter_by(is_approved=False).count() == 1
    assert db_request.db.query(OrganizationApplication).filter_by(is_approved=None).count() == 1