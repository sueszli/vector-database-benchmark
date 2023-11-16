import pytest
from ckan import model
from ckan.tests import helpers, factories

@pytest.mark.usefixtures(u'non_clean_db')
class TestPackageExtra(object):

    def test_create_extras(self):
        if False:
            print('Hello World!')
        pkg = model.Package(name=factories.Dataset.stub().name)
        extra1 = model.PackageExtra(key=u'subject', value=u'science')
        pkg._extras[u'subject'] = extra1
        pkg.extras[u'accuracy'] = u'metre'
        model.Session.add_all([pkg])
        model.Session.commit()
        model.Session.remove()
        pkg = model.Package.by_name(pkg.name)
        assert pkg.extras == {u'subject': u'science', u'accuracy': u'metre'}

    def test_delete_extras(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = factories.Dataset(extras=[{u'key': u'subject', u'value': u'science'}, {u'key': u'accuracy', u'value': u'metre'}])
        pkg = model.Package.by_name(dataset[u'name'])
        del pkg.extras[u'subject']
        model.Session.commit()
        model.Session.remove()
        pkg = model.Package.by_name(dataset[u'name'])
        assert pkg.extras == {u'accuracy': u'metre'}

    def test_extras_list(self):
        if False:
            i = 10
            return i + 15
        extras = [{u'key': u'subject', u'value': u'science'}, {u'key': u'accuracy', u'value': u'metre'}, {u'key': u'sample_years', u'value': u'2012-2013'}]
        dataset = factories.Dataset(extras=extras)
        extras = extras[1:]
        helpers.call_action(u'package_patch', id=dataset['id'], extras=extras)
        factories.Dataset(extras=[{u'key': u'foo', u'value': u'bar'}])
        pkg = model.Package.by_name(dataset[u'name'])
        assert isinstance(pkg.extras_list[0], model.PackageExtra)
        assert set([(pe.package_id, pe.key, pe.value, pe.state) for pe in pkg.extras_list]) == set([(dataset['id'], u'accuracy', u'metre', u'active'), (dataset['id'], u'sample_years', u'2012-2013', u'active')])