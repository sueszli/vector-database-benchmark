import pytest
from unittest import mock
import json
from awx.main.models import ActivityStream, Organization, JobTemplate, Credential, CredentialType, Inventory, InventorySource, Project, User
from awx.main.utils import model_to_dict, model_instance_diff
from awx.main.utils.common import get_allowed_fields
from awx.main.signals import model_serializer_mapping
from django.contrib.auth.models import AnonymousUser
from crum import impersonate

class TestImplicitRolesOmitted:
    """
    Test that there is exactly 1 "create" entry in the activity stream for
    common items in the system.
    These tests will fail if `rbac_activity_stream` creates
    false-positive entries.
    """

    @pytest.mark.django_db
    def test_activity_stream_create_organization(self):
        if False:
            i = 10
            return i + 15
        Organization.objects.create(name='test-organization2')
        qs = ActivityStream.objects.filter(organization__isnull=False)
        assert qs.count() == 1
        assert qs[0].operation == 'create'

    @pytest.mark.django_db
    def test_activity_stream_delete_organization(self):
        if False:
            for i in range(10):
                print('nop')
        org = Organization.objects.create(name='gYSlNSOFEW')
        org.delete()
        qs = ActivityStream.objects.filter(changes__icontains='gYSlNSOFEW')
        assert qs.count() == 2
        assert qs[1].operation == 'delete'

    @pytest.mark.django_db
    def test_activity_stream_create_JT(self, project, inventory):
        if False:
            i = 10
            return i + 15
        JobTemplate.objects.create(name='test-jt', project=project, inventory=inventory)
        qs = ActivityStream.objects.filter(job_template__isnull=False)
        assert qs.count() == 1
        assert qs[0].operation == 'create'

    @pytest.mark.django_db
    def test_activity_stream_create_inventory(self, organization):
        if False:
            i = 10
            return i + 15
        organization.inventories.create(name='test-inv')
        qs = ActivityStream.objects.filter(inventory__isnull=False)
        assert qs.count() == 1
        assert qs[0].operation == 'create'

    @pytest.mark.django_db
    def test_activity_stream_create_credential(self, organization):
        if False:
            print('Hello World!')
        organization.inventories.create(name='test-inv')
        qs = ActivityStream.objects.filter(inventory__isnull=False)
        assert qs.count() == 1
        assert qs[0].operation == 'create'

@pytest.mark.django_db
class TestRolesAssociationEntries:
    """
    Test that non-implicit role associations have a corresponding
    activity stream entry.
    These tests will fail if `rbac_activity_stream` skipping logic
    in signals is wrong.
    """

    def test_non_implicit_associations_are_recorded(self, project):
        if False:
            for i in range(10):
                print('nop')
        org2 = Organization.objects.create(name='test-organization2')
        for i in range(2):
            project.admin_role.parents.add(org2.admin_role)
            assert ActivityStream.objects.filter(role=org2.admin_role, organization=org2, project=project).count() == 1, 'In loop %s' % i

    def test_model_associations_are_recorded(self, organization):
        if False:
            while True:
                i = 10
        proj1 = Project.objects.create(name='proj1', organization=organization)
        proj2 = Project.objects.create(name='proj2', organization=organization)
        proj2.use_role.parents.add(proj1.admin_role)
        assert ActivityStream.objects.filter(role=proj1.admin_role, project=proj2).count() == 1

    @pytest.mark.parametrize('value', [True, False])
    def test_auditor_is_recorded(self, post, value):
        if False:
            return 10
        u = User.objects.create(username='foouser')
        assert not u.is_system_auditor
        u.is_system_auditor = value
        u = User.objects.get(pk=u.pk)
        assert u.is_system_auditor == value
        entry_qs = ActivityStream.objects.filter(user=u)
        if value:
            assert len(entry_qs) == 2
        else:
            assert len(entry_qs) == 1
        assert 'is_system_auditor' not in json.loads(entry_qs[0].changes)
        if value:
            auditor_changes = json.loads(entry_qs[1].changes)
            assert auditor_changes['object2'] == 'user'
            assert auditor_changes['object2_pk'] == u.pk

    def test_user_no_op_api(self, system_auditor):
        if False:
            return 10
        as_ct = ActivityStream.objects.count()
        system_auditor.is_system_auditor = True
        assert ActivityStream.objects.count() == as_ct

@pytest.fixture
def somecloud_type():
    if False:
        while True:
            i = 10
    return CredentialType.objects.create(kind='cloud', name='SomeCloud', managed=False, inputs={'fields': [{'id': 'api_token', 'label': 'API Token', 'type': 'string', 'secret': True}]}, injectors={'env': {'MY_CLOUD_API_TOKEN': '{{api_token.foo()}}'}})

@pytest.mark.django_db
class TestCredentialModels:
    """
    Assure that core elements of activity stream feature are working
    """

    def test_create_credential_type(self, somecloud_type):
        if False:
            return 10
        assert ActivityStream.objects.filter(credential_type=somecloud_type).count() == 1
        entry = ActivityStream.objects.filter(credential_type=somecloud_type)[0]
        assert entry.operation == 'create'

    def test_credential_hidden_information(self, somecloud_type):
        if False:
            return 10
        cred = Credential.objects.create(credential_type=somecloud_type, inputs={'api_token': 'ABC123'})
        entry = ActivityStream.objects.filter(credential=cred)[0]
        assert entry.operation == 'create'
        assert json.loads(entry.changes)['inputs'] == 'hidden'

@pytest.mark.django_db
class TestUserModels:

    def test_user_hidden_information(self, alice):
        if False:
            print('Hello World!')
        entry = ActivityStream.objects.filter(user=alice)[0]
        assert entry.operation == 'create'
        assert json.loads(entry.changes)['password'] == 'hidden'

@pytest.mark.django_db
def test_missing_related_on_delete(inventory_source):
    if False:
        print('Hello World!')
    old_is = InventorySource.objects.get(name=inventory_source.name)
    inventory_source.inventory.delete()
    d = model_to_dict(old_is, serializer_mapping=model_serializer_mapping())
    assert d['inventory'] == '<missing inventory source>-{}'.format(old_is.inventory_id)

@pytest.mark.django_db
def test_activity_stream_actor(admin_user):
    if False:
        i = 10
        return i + 15
    with impersonate(admin_user):
        o = Organization.objects.create(name='test organization')
    entry = o.activitystream_set.get(operation='create')
    assert entry.actor == admin_user

@pytest.mark.django_db
def test_anon_user_action():
    if False:
        while True:
            i = 10
    with mock.patch('awx.main.signals.get_current_user') as u_mock:
        u_mock.return_value = AnonymousUser()
        inv = Inventory.objects.create(name='ainventory')
    entry = inv.activitystream_set.filter(operation='create').first()
    assert not entry.actor

@pytest.mark.django_db
def test_activity_stream_deleted_actor(alice, bob):
    if False:
        return 10
    alice.first_name = 'Alice'
    alice.last_name = 'Doe'
    alice.save()
    with impersonate(alice):
        o = Organization.objects.create(name='test organization')
    entry = o.activitystream_set.get(operation='create')
    assert entry.actor == alice
    alice.delete()
    entry = o.activitystream_set.get(operation='create')
    assert entry.actor is None
    deleted = entry.deleted_actor
    assert deleted['username'] == 'alice'
    assert deleted['first_name'] == 'Alice'
    assert deleted['last_name'] == 'Doe'
    entry.actor = bob
    entry.save(update_fields=['actor'])
    deleted = entry.deleted_actor
    entry = ActivityStream.objects.get(id=entry.pk)
    assert entry.deleted_actor['username'] == 'bob'

@pytest.mark.django_db
def test_modified_not_allowed_field(somecloud_type):
    if False:
        return 10
    '\n    If this test fails, that means that read-only fields are showing\n    up in the activity stream serialization of an instance.\n\n    That _probably_ means that you just connected a new model to the\n    activity_stream_registrar, but did not add its serializer to\n    the model->serializer mapping.\n    '
    from awx.main.registrar import activity_stream_registrar
    for Model in activity_stream_registrar.models:
        assert 'modified' not in get_allowed_fields(Model(), model_serializer_mapping()), Model

@pytest.mark.django_db
def test_survey_spec_create_entry(job_template, survey_spec_factory):
    if False:
        for i in range(10):
            print('nop')
    start_count = job_template.activitystream_set.count()
    job_template.survey_spec = survey_spec_factory('foo')
    job_template.save()
    assert job_template.activitystream_set.count() == start_count + 1

@pytest.mark.django_db
def test_survey_create_diff(job_template, survey_spec_factory):
    if False:
        return 10
    old = JobTemplate.objects.get(pk=job_template.pk)
    job_template.survey_spec = survey_spec_factory('foo')
    (before, after) = model_instance_diff(old, job_template, model_serializer_mapping())['survey_spec']
    assert before == '{}'
    assert json.loads(after) == survey_spec_factory('foo')

@pytest.mark.django_db
def test_saved_passwords_hidden_activity(workflow_job_template, job_template_with_survey_passwords):
    if False:
        print('Hello World!')
    node_with_passwords = workflow_job_template.workflow_nodes.create(unified_job_template=job_template_with_survey_passwords, extra_data={'bbbb': '$encrypted$fooooo'}, survey_passwords={'bbbb': '$encrypted$'})
    node_with_passwords.delete()
    entry = ActivityStream.objects.order_by('timestamp').last()
    changes = json.loads(entry.changes)
    assert 'survey_passwords' not in changes
    assert json.loads(changes['extra_data'])['bbbb'] == '$encrypted$'

@pytest.mark.django_db
def test_cluster_node_recorded(inventory, project):
    if False:
        for i in range(10):
            print('nop')
    jt = JobTemplate.objects.create(name='testjt', inventory=inventory, project=project)
    with mock.patch('awx.main.models.activity_stream.settings.CLUSTER_HOST_ID', 'foo_host'):
        job = jt.create_unified_job()
    entry = ActivityStream.objects.filter(job=job).first()
    assert entry.action_node == 'foo_host'

@pytest.mark.django_db
def test_cluster_node_long_node_name(inventory, project):
    if False:
        print('Hello World!')
    jt = JobTemplate.objects.create(name='testjt', inventory=inventory, project=project)
    with mock.patch('awx.main.models.activity_stream.settings.CLUSTER_HOST_ID', 'f' * 700):
        job = jt.create_unified_job()
    entry = ActivityStream.objects.filter(job=job).first()
    assert entry.action_node.startswith('ffffff')

@pytest.mark.django_db
def test_credential_defaults_idempotency():
    if False:
        i = 10
        return i + 15
    CredentialType.setup_tower_managed_defaults()
    old_inputs = CredentialType.objects.get(name='Red Hat Ansible Automation Platform', kind='cloud').inputs
    prior_count = ActivityStream.objects.count()
    CredentialType.setup_tower_managed_defaults()
    assert CredentialType.objects.get(name='Red Hat Ansible Automation Platform', kind='cloud').inputs == old_inputs
    assert ActivityStream.objects.count() == prior_count