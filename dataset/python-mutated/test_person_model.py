from uuid import uuid4
from posthog.client import sync_execute
from posthog.models import Person, PersonDistinctId
from posthog.models.event.util import create_event
from posthog.models.person.util import delete_person
from posthog.test.base import BaseTest

def _create_event(**kwargs):
    if False:
        return 10
    pk = uuid4()
    kwargs.update({'event_uuid': pk})
    create_event(**kwargs)

class TestPerson(BaseTest):

    def test_person_is_identified(self):
        if False:
            return 10
        person_identified = Person.objects.create(team=self.team, is_identified=True)
        person_anonymous = Person.objects.create(team=self.team)
        self.assertEqual(person_identified.is_identified, True)
        self.assertEqual(person_anonymous.is_identified, False)

    def test_delete_person(self):
        if False:
            while True:
                i = 10
        person = Person.objects.create(team=self.team, version=15)
        delete_person(person, sync=True)
        ch_persons = sync_execute('SELECT toString(id), version, is_deleted, properties FROM person FINAL WHERE team_id = %(team_id)s and id = %(uuid)s', {'team_id': self.team.pk, 'uuid': person.uuid})
        self.assertEqual(ch_persons, [(str(person.uuid), 115, 1, '{}')])

    def test_delete_ch_distinct_ids(self):
        if False:
            i = 10
            return i + 15
        person = Person.objects.create(team=self.team)
        PersonDistinctId.objects.create(team=self.team, person=person, distinct_id='distinct_id1', version=15)
        ch_distinct_ids = sync_execute('SELECT is_deleted FROM person_distinct_id2 FINAL WHERE team_id = %(team_id)s and distinct_id = %(distinct_id)s', {'team_id': self.team.pk, 'distinct_id': 'distinct_id1'})
        self.assertEqual(ch_distinct_ids, [(0,)])
        delete_person(person, sync=True)
        ch_distinct_ids = sync_execute('SELECT toString(person_id), version, is_deleted FROM person_distinct_id2 FINAL WHERE team_id = %(team_id)s and distinct_id = %(distinct_id)s', {'team_id': self.team.pk, 'distinct_id': 'distinct_id1'})
        self.assertEqual(ch_distinct_ids, [(str(person.uuid), 115, 1)])