import contextlib
import datetime as dt
from threading import Event, Thread
from uuid import uuid4
import pytest
from django.db.utils import DEFAULT_DB_ALIAS, ConnectionHandler, IntegrityError
from posthog.api.test.test_organization import create_organization
from posthog.api.test.test_team import create_team
from posthog.models import Person, PersonOverride, PersonOverrideMapping, Team

@pytest.fixture
def organization():
    if False:
        for i in range(10):
            print('nop')
    organization = create_organization(name='test')
    yield organization
    organization.delete()

@pytest.fixture
def team(organization):
    if False:
        for i in range(10):
            print('nop')
    team = create_team(organization=organization)
    yield team
    team.delete()

@pytest.fixture
def people(team):
    if False:
        return 10
    old_person_uuid = uuid4()
    override_person_uuid = uuid4()
    new_override_person_uuid = uuid4()
    p1 = Person.objects.create(uuid=old_person_uuid, team=team)
    p2 = Person.objects.create(uuid=override_person_uuid, team=team)
    p3 = Person.objects.create(uuid=new_override_person_uuid, team=team)
    yield (p1, p2, p3)
    p1.delete()
    p2.delete()
    p3.delete()

@pytest.fixture
def oldest_event():
    if False:
        i = 10
        return i + 15
    return dt.datetime.now(dt.timezone.utc)

@pytest.mark.django_db(transaction=True)
def test_person_override_disallows_same_old_person_id(team, oldest_event):
    if False:
        return 10
    'Test a new old_person_id cannot match an existing old_person_id.\n\n    This is enforced by a UNIQUE constraint on (team_id, old_person_id)\n    '
    old_person_id = uuid4()
    override_person_id = uuid4()
    new_override_person_id = uuid4()
    old_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=old_person_id)
    override_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=override_person_id)
    person_override = PersonOverride.objects.create(team=team, old_person_id=old_mapping, override_person_id=override_mapping, oldest_event=oldest_event, version=1)
    person_override.save()
    assert person_override.old_person_id == old_mapping
    assert person_override.override_person_id == override_mapping
    new_override_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=new_override_person_id)
    with pytest.raises(IntegrityError):
        PersonOverride.objects.create(team=team, old_person_id=old_mapping, override_person_id=new_override_mapping, oldest_event=oldest_event, version=1).save()

@pytest.mark.django_db(transaction=True)
def test_person_override_same_old_person_id_in_different_teams(organization, team, oldest_event):
    if False:
        print('Hello World!')
    'Test a new old_person_id can match an existing from a different team.'
    old_person_id = uuid4()
    override_person_id = uuid4()
    new_team = Team.objects.create(organization=organization, api_token='a different token')
    old_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=old_person_id)
    override_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=override_person_id)
    p1 = PersonOverride.objects.create(team=team, old_person_id=old_mapping, override_person_id=override_mapping, oldest_event=oldest_event, version=1)
    p1.save()
    assert p1.old_person_id == old_mapping
    assert p1.override_person_id == override_mapping
    new_team_old_mapping = PersonOverrideMapping.objects.create(team_id=new_team.id, uuid=old_person_id)
    new_team_override_mapping = PersonOverrideMapping.objects.create(team_id=new_team.id, uuid=override_person_id)
    p2 = PersonOverride.objects.create(team=new_team, old_person_id=new_team_old_mapping, override_person_id=new_team_override_mapping, oldest_event=oldest_event, version=1)
    p2.save()
    assert p1.old_person_id.uuid == p2.old_person_id.uuid
    assert p1.override_person_id.uuid == p2.override_person_id.uuid
    assert p1.old_person_id.id != p2.old_person_id.id
    assert p1.override_person_id.id != p2.override_person_id.id
    assert p1.team != p2.team

@pytest.mark.django_db(transaction=True)
def test_person_override_disallows_override_person_id_as_old_person_id(team, oldest_event):
    if False:
        while True:
            i = 10
    'Test a new old_person_id cannot match an existing override_person_id.\n\n    We re-use the override_person_id from the first model created as the old_person_id\n    of the second model. We expect an exception on saving this second model.\n    '
    old_person_id = uuid4()
    override_person_id = uuid4()
    new_override_person_id = uuid4()
    old_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=old_person_id)
    override_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=override_person_id)
    person_override = PersonOverride.objects.create(team=team, old_person_id=old_mapping, override_person_id=override_mapping, oldest_event=oldest_event, version=1)
    person_override.save()
    assert person_override.old_person_id == old_mapping
    assert person_override.override_person_id == override_mapping
    new_override_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=new_override_person_id)
    with pytest.raises(IntegrityError):
        PersonOverride.objects.create(team=team, old_person_id=override_mapping, override_person_id=new_override_mapping, oldest_event=oldest_event, version=1).save()

@pytest.mark.django_db(transaction=True)
def test_person_override_allows_override_person_id_as_old_person_id_in_different_teams(team, organization, oldest_event):
    if False:
        for i in range(10):
            print('nop')
    'Test a new old_person_id can match an override in a different team.'
    old_person_id = uuid4()
    override_person_id = uuid4()
    new_override_person_id = uuid4()
    new_team = Team.objects.create(organization=organization, api_token='a much different token')
    old_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=old_person_id)
    override_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=override_person_id)
    p1 = PersonOverride.objects.create(team=team, old_person_id=old_mapping, override_person_id=override_mapping, oldest_event=oldest_event, version=1)
    p1.save()
    assert p1.old_person_id == old_mapping
    assert p1.override_person_id == override_mapping
    new_team_old_mapping = PersonOverrideMapping.objects.create(team_id=new_team.id, uuid=override_person_id)
    new_team_override_mapping = PersonOverrideMapping.objects.create(team_id=new_team.id, uuid=new_override_person_id)
    p2 = PersonOverride.objects.create(team=new_team, old_person_id=new_team_old_mapping, override_person_id=new_team_override_mapping, oldest_event=oldest_event, version=1)
    p2.save()
    assert p1.override_person_id.uuid == p2.old_person_id.uuid
    assert p2.override_person_id == new_team_override_mapping
    assert p1.team != p2.team

@pytest.mark.django_db(transaction=True)
def test_person_override_disallows_old_person_id_as_override_person_id(team, oldest_event):
    if False:
        print('Hello World!')
    'Test a new override_person_id cannot match an existing old_person_id.\n\n    We re-use the old_person_id from the first model created as the override_person_id\n    of the second model. We expect an exception on saving this second model.\n    '
    old_person_id = uuid4()
    override_person_id = uuid4()
    new_old_person_id = uuid4()
    old_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=old_person_id)
    old_mapping.save()
    override_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=override_person_id)
    override_mapping.save()
    person_override = PersonOverride.objects.create(team=team, old_person_id=old_mapping, override_person_id=override_mapping, oldest_event=oldest_event, version=1)
    person_override.save()
    assert person_override.old_person_id == old_mapping
    assert person_override.override_person_id == override_mapping
    new_old_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=new_old_person_id)
    new_old_mapping.save()
    with pytest.raises(IntegrityError):
        p = PersonOverride.objects.create(team=team, old_person_id=new_old_mapping, override_person_id=old_mapping, oldest_event=oldest_event, version=1)
        p.save()

@pytest.mark.django_db(transaction=True)
def test_person_override_old_person_id_as_override_person_id_in_different_teams(organization, team, oldest_event):
    if False:
        i = 10
        return i + 15
    'Test a new override_person_id can match an old in a different team.'
    old_person_id = uuid4()
    override_person_id = uuid4()
    new_old_person_id = uuid4()
    new_team = Team.objects.create(organization=organization, api_token='a significantly different token')
    old_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=old_person_id)
    override_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=override_person_id)
    p1 = PersonOverride.objects.create(team=team, old_person_id=old_mapping, override_person_id=override_mapping, oldest_event=oldest_event, version=1)
    p1.save()
    assert p1.old_person_id == old_mapping
    assert p1.override_person_id == override_mapping
    new_old_mapping = PersonOverrideMapping.objects.create(team_id=new_team.id, uuid=new_old_person_id)
    new_override_mapping = PersonOverrideMapping.objects.create(team_id=new_team.id, uuid=old_mapping.uuid)
    p2 = PersonOverride.objects.create(team=new_team, old_person_id=new_old_mapping, override_person_id=new_override_mapping, oldest_event=oldest_event, version=1)
    p2.save()
    assert p1.old_person_id.uuid == p2.override_person_id.uuid
    assert p1.old_person_id.team_id == p1.override_person_id.team_id
    assert p2.old_person_id == new_old_mapping
    assert p1.team != p2.team

@pytest.mark.django_db(transaction=True)
def test_person_override_allows_duplicate_override_person_id(team, oldest_event):
    if False:
        while True:
            i = 10
    'Test duplicate override_person_ids with different old_person_ids are allowed.'
    override_person_id = uuid4()
    n_person_overrides = 2
    created = []
    override_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=override_person_id)
    for _ in range(n_person_overrides):
        old_person_id = uuid4()
        old_mapping = PersonOverrideMapping.objects.create(team_id=team.id, uuid=old_person_id)
        person_override = PersonOverride.objects.create(team=team, old_person_id=old_mapping, override_person_id=override_mapping, oldest_event=oldest_event, version=1)
        person_override.save()
        created.append(person_override)
    assert all((p.override_person_id == override_mapping for p in created))
    assert len(set((p.old_person_id.uuid for p in created))) == n_person_overrides

@contextlib.contextmanager
def create_connection(alias=DEFAULT_DB_ALIAS):
    if False:
        return 10
    connection = ConnectionHandler().create_connection(alias)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET lock_timeout TO '10s'")
            try:
                yield cursor
            finally:
                cursor.execute('ROLLBACK')
                cursor.close()
    finally:
        connection.close()

def _merge_people(team, cursor, old_person_uuid, override_person_uuid, oldest_event, can_lock_event=None, done_event=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Merge two people together, using the override_person_id as the canonical\n    person.\n\n    This function is meant to be run in a separate thread, so that we can test\n    that the transaction is rolled back if there is a conflict.\n    This mimics how we expect the code to do person merges, i.e. in a transaction\n    that deletes the old person, adds old person->override person override and updates\n    all old person as override person rows to now point to the new override person.\n    Note that we don't actually handle the merge on the posthog_person table,\n    but rather simply DELETE the record associated with `old_person_id`. It may\n    be that we remove the implmentation of deleting merged persons, in which\n    case we'll need to update the constraint to also include e.g. the\n    `is_deleted` flag we may add.\n    "
    cursor.execute('\n            DELETE FROM\n                posthog_person\n            WHERE\n                uuid = %(old_person_uuid)s\n                AND team_id = %(team_id)s;\n        ', {'team_id': team.id, 'old_person_uuid': old_person_uuid})
    if can_lock_event is not None:
        can_lock_event.wait(10)
    cursor.execute('\n            WITH insert_id AS (\n                INSERT INTO posthog_personoverridemapping(\n                    team_id,\n                    uuid\n                )\n                VALUES (\n                    %(team_id)s,\n                    %(old_person_uuid)s\n                )\n                ON CONFLICT("team_id", "uuid") DO NOTHING\n                RETURNING id\n            )\n            SELECT * FROM insert_id\n            UNION ALL\n            -- ON CONFLICT nothing is returned, so we get the id here.\n            -- Fear not, the constraints on personoverride will handle any inconsistencies.\n            -- This mapping table is really nothing more than a mapping.\n            SELECT id\n            FROM posthog_personoverridemapping\n            WHERE uuid = %(old_person_uuid)s\n        ', {'team_id': team.id, 'old_person_uuid': old_person_uuid})
    old_person_id = cursor.fetchone()[0]
    cursor.execute('\n            WITH insert_id AS (\n                INSERT INTO posthog_personoverridemapping(\n                    team_id,\n                    uuid\n                )\n                VALUES (\n                    %(team_id)s,\n                    %(override_person_uuid)s\n                )\n                ON CONFLICT("team_id", "uuid") DO NOTHING\n                RETURNING id\n            )\n            SELECT * FROM insert_id\n            UNION ALL\n            SELECT id\n            FROM posthog_personoverridemapping\n            WHERE uuid = %(override_person_uuid)s\n        ', {'team_id': team.id, 'override_person_uuid': override_person_uuid})
    override_person_id = cursor.fetchone()[0]
    cursor.execute('\n            INSERT INTO posthog_personoverride(\n                team_id,\n                old_person_id,\n                override_person_id,\n                oldest_event,\n                version\n            )\n            VALUES (\n                %(team_id)s,\n                %(old_person_id)s,\n                %(override_person_id)s,\n                %(oldest_event)s,\n                1\n            );\n            UPDATE\n                posthog_personoverride\n            SET\n                override_person_id = %(override_person_id)s,\n                version = version + 1\n            WHERE override_person_id = %(old_person_id)s\n                  AND team_id = %(team_id)s;\n        ', {'team_id': team.id, 'old_person_id': old_person_id, 'override_person_id': override_person_id, 'oldest_event': oldest_event})
    if done_event is not None:
        done_event.set()
"\nConcurrency tests for person overrides table\nGoal: verify that we don't end up in a situation with the same uuid is both\nan old person id and an override person id\n- there are two cases that we want to check for\n    - concurrent merges\n    - concurrent merge and person deletion\nIn both cases one of the transactions will wait on the lock,\nso they can only complete in one order (which is tested below).\nNote that to test the race condition scenario we need to:\n    1. create multiple concurrent transactions, such that we can verify\n    constraints are enforced at COMMIT time.\n    2. enable transactions for the Django test. This is more so we can see data\n    from the main Django PostgreSQL connection session in the other\n    concurrent transactions. Not 100% required but makes things a little\n    easier to write.\n"

@pytest.mark.django_db(transaction=True)
def test_person_override_merge(people, team, oldest_event):
    if False:
        for i in range(10):
            print('nop')
    'Verify our merge function work as expected.'
    (old_person, override_person, _) = people
    with create_connection() as merge_cursor:
        merge_cursor.execute('BEGIN')
        _merge_people(team, merge_cursor, old_person.uuid, override_person.uuid, oldest_event)
        merge_cursor.execute('COMMIT')
    assert [_[0] for _ in PersonOverrideMapping.objects.all().values_list('uuid')] == [old_person.uuid, override_person.uuid]
    old_person_id = PersonOverrideMapping.objects.filter(uuid=old_person.uuid).all()[0].id
    override_person_id = PersonOverrideMapping.objects.filter(uuid=override_person.uuid).all()[0].id
    assert list(PersonOverride.objects.all().values_list('old_person_id', 'override_person_id')) == [(old_person_id, override_person_id)]

@pytest.mark.django_db(transaction=True)
def test_person_override_allow_consecutive_merges(people, team, oldest_event):
    if False:
        while True:
            i = 10
    'Verify our merge function works as expected when called consecutively.'
    (old_person, override_person, new_override_person) = people
    with create_connection() as first_cursor:
        first_cursor.execute('BEGIN')
        _merge_people(team, first_cursor, old_person.uuid, override_person.uuid, oldest_event)
        first_cursor.execute('COMMIT')
    with create_connection() as second_cursor:
        second_cursor.execute('BEGIN')
        _merge_people(team, second_cursor, override_person.uuid, new_override_person.uuid, oldest_event)
        second_cursor.execute('COMMIT')
    assert [_[0] for _ in PersonOverrideMapping.objects.all().values_list('uuid')] == [old_person.uuid, override_person.uuid, new_override_person.uuid]
    old_person_id = PersonOverrideMapping.objects.filter(uuid=old_person.uuid).all()[0].id
    override_person_id = PersonOverrideMapping.objects.filter(uuid=override_person.uuid).all()[0].id
    new_override_person_id = PersonOverrideMapping.objects.filter(uuid=new_override_person.uuid).all()[0].id
    mappings = list(PersonOverride.objects.all().values_list('old_person_id', 'override_person_id'))
    assert sorted(mappings) == sorted([(override_person_id, new_override_person_id), (old_person_id, new_override_person_id)]), f'mappings={mappings!r} old_person_id={old_person_id!r}, override_person_id={override_person_id!r}, new_override_person_id={new_override_person_id!r}'

@pytest.mark.django_db(transaction=True)
def test_person_override_disallows_concurrent_merge(people, team, oldest_event):
    if False:
        while True:
            i = 10
    'Test concurrent merges.\n\n    Running two merges:\n    A: old_person -> override_person\n    B: override_person -> new_override_person\n\n    Both merges are run in their own transactions, but for this test the B merge will be\n    committed first, before A has had a chance to lock the tables.\n\n    Then A should raise an exception, as it now violates an integrity constraint (trying to\n    use override_person_id that already exists as old_person_id)\n    '
    (old_person, override_person, new_override_person) = people
    with create_connection() as first_cursor, create_connection() as second_cursor:
        first_cursor.execute('BEGIN')
        second_cursor.execute('BEGIN')
        can_lock_event = Event()
        done_t1_event = Event()
        done_t2_event = Event()
        t1 = Thread(target=_merge_people, args=(team, first_cursor, old_person.uuid, override_person.uuid, oldest_event), kwargs={'can_lock_event': can_lock_event, 'done_event': done_t1_event})
        t2 = Thread(target=_merge_people, args=(team, second_cursor, override_person.uuid, new_override_person.uuid, oldest_event), kwargs={'done_event': done_t2_event})
        t1.start()
        t2.start()
        done_t2_event.wait(10)
        second_cursor.execute('COMMIT')
        with pytest.raises(IntegrityError):
            can_lock_event.set()
            done_t1_event.wait(10)
            first_cursor.execute('COMMIT')
    assert [_[0] for _ in PersonOverrideMapping.objects.all().values_list('uuid')] == [override_person.uuid, new_override_person.uuid]
    override_person_id = PersonOverrideMapping.objects.filter(uuid=override_person.uuid).all()[0].id
    new_override_person_id = PersonOverrideMapping.objects.filter(uuid=new_override_person.uuid).all()[0].id
    assert list(PersonOverride.objects.all().values_list('old_person_id', 'override_person_id')) == [(override_person_id, new_override_person_id)]

@pytest.mark.django_db(transaction=True)
def test_person_override_disallows_concurrent_merge_different_order(people, team, oldest_event):
    if False:
        for i in range(10):
            print('nop')
    'Test concurrent merges but in a valid consecutive order.\n\n    Running two merges:\n    A: old_person -> override_person\n    B: override_person -> new_override_person\n\n    Both merges are run in their own transactions, and in this test the A merge will be\n    committed first, before B has had a chance to lock the tables.\n\n    Then both should succeed as these merges are compatible: B will simply update\n    override_person to new_override_person when it is allowed to run.\n    This is just the concurrent version of the test scenario from test_person_override_allow_consecutive_merges.\n    '
    (old_person, override_person, new_override_person) = people
    with create_connection() as first_cursor, create_connection() as second_cursor:
        first_cursor.execute('BEGIN')
        second_cursor.execute('BEGIN')
        can_lock_event = Event()
        done_t1_event = Event()
        done_t2_event = Event()
        t1 = Thread(target=_merge_people, args=(team, first_cursor, old_person.uuid, override_person.uuid, oldest_event), kwargs={'done_event': done_t1_event})
        t2 = Thread(target=_merge_people, args=(team, second_cursor, override_person.uuid, new_override_person.uuid, oldest_event), kwargs={'can_lock_event': can_lock_event, 'done_event': done_t2_event})
        t1.start()
        t2.start()
        done_t1_event.wait(10)
        first_cursor.execute('COMMIT')
        can_lock_event.set()
        done_t2_event.wait(10)
        second_cursor.execute('COMMIT')
    assert [_[0] for _ in PersonOverrideMapping.objects.all().values_list('uuid')] == [old_person.uuid, override_person.uuid, new_override_person.uuid]
    old_person_id = PersonOverrideMapping.objects.filter(uuid=old_person.uuid).all()[0].id
    override_person_id = PersonOverrideMapping.objects.filter(uuid=override_person.uuid).all()[0].id
    new_override_person_id = PersonOverrideMapping.objects.filter(uuid=new_override_person.uuid).all()[0].id
    assert list(PersonOverride.objects.all().values_list('old_person_id', 'override_person_id', 'version')) == [(override_person_id, new_override_person_id, 1), (old_person_id, new_override_person_id, 2)]