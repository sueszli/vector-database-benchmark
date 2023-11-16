import pretend
from warehouse.admin.views import includes
from ....common.db.packaging import ProjectFactory

def test_administer_project_include_returns_project(db_request):
    if False:
        i = 10
        return i + 15
    project = ProjectFactory.create()
    db_request.matchdict = {'project_name': project.name}
    assert includes.administer_project_include(db_request) == {'project': project, 'prohibited': None, 'project_name': project.name}

def test_administer_user_include_returns_user():
    if False:
        i = 10
        return i + 15
    user = pretend.stub()
    assert includes.administer_user_include(user, pretend.stub()) == {'user': user}