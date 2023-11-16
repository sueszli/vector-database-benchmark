import copy
import pytest
import ckan.model as model
from ckan.tests.helpers import call_action
from ckanext.activity.model import activity_dict_save, activity as activity_model, Activity

@pytest.mark.ckan_config('ckan.plugins', 'activity')
@pytest.mark.usefixtures('with_plugins', 'clean_db')
class TestActivity(object):

    def test_include_data(self, package, user, activity_factory):
        if False:
            return 10
        activity = activity_factory(user_id=user['id'], object_id=package['id'], activity_type='new package', data={'package': copy.deepcopy(package), 'actor': 'Mr Someone'})
        activity_obj = Activity.get(activity['id'])
        context = {'model': model, 'session': model.Session}
        dictized = activity_model.activity_dictize(activity_obj, context)
        assert dictized['user_id'] == user['id']
        assert dictized['activity_type'] == 'new package'
        assert dictized['data']['package']['title'] == package['title']
        assert dictized['data']['package']['id'] == package['id']
        assert dictized['data']['actor'] == 'Mr Someone'

    def test_activity_save(self, user):
        if False:
            return 10
        context = {'model': model, 'session': model.Session}
        sent = {'user_id': user['id'], 'object_id': user['id'], 'activity_type': 'changed user'}
        activity_dict_save(sent, context)
        model.Session.commit()
        got = call_action('user_activity_list', context, id=user['id'])[0]
        assert got['user_id'] == sent['user_id']
        assert got['object_id'] == sent['object_id']
        assert got['activity_type'] == sent['activity_type']
        assert got['id']
        assert got['timestamp']