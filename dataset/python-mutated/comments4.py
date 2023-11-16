from com.my_lovely_company.my_lovely_team.my_lovely_project.my_lovely_component import MyLovelyCompanyTeamProjectComponent
from com.my_lovely_company.my_lovely_team.my_lovely_project.my_lovely_component import MyLovelyCompanyTeamProjectComponent as component

class C:

    @pytest.mark.parametrize(('post_data', 'message'), [({}, 'None is an invalid value for Metadata-Version. Error: This field is required. see https://packaging.python.org/specifications/core-metadata'), ({'metadata_version': '-1'}, "'-1' is an invalid value for Metadata-Version. Error: Unknown Metadata Version see https://packaging.python.org/specifications/core-metadata"), ({'metadata_version': '1.2'}, "'' is an invalid value for Name. Error: This field is required. see https://packaging.python.org/specifications/core-metadata"), ({'metadata_version': '1.2', 'name': 'foo-'}, "'foo-' is an invalid value for Name. Error: Must start and end with a letter or numeral and contain only ascii numeric and '.', '_' and '-'. see https://packaging.python.org/specifications/core-metadata"), ({'metadata_version': '1.2', 'name': 'example'}, "'' is an invalid value for Version. Error: This field is required. see https://packaging.python.org/specifications/core-metadata"), ({'metadata_version': '1.2', 'name': 'example', 'version': 'dog'}, "'dog' is an invalid value for Version. Error: Must start and end with a letter or numeral and contain only ascii numeric and '.', '_' and '-'. see https://packaging.python.org/specifications/core-metadata")])
    def test_fails_invalid_post_data(self, pyramid_config, db_request, post_data, message):
        if False:
            i = 10
            return i + 15
        pyramid_config.testing_securitypolicy(userid=1)
        db_request.POST = MultiDict(post_data)

def foo(list_a, list_b):
    if False:
        while True:
            i = 10
    results = User.query.filter(User.foo == 'bar').filter(db.or_(User.field_a.astext.in_(list_a), User.field_b.astext.in_(list_b))).filter(User.xyz.is_(None)).filter(db.not_(User.is_pending.astext.cast(db.Boolean).is_(True))).order_by(User.created_at.desc()).with_for_update(key_share=True).all()
    return results

def foo2(list_a, list_b):
    if False:
        i = 10
        return i + 15
    return User.query.filter(User.foo == 'bar').filter(db.or_(User.field_a.astext.in_(list_a), User.field_b.astext.in_(list_b))).filter(User.xyz.is_(None))

def foo3(list_a, list_b):
    if False:
        while True:
            i = 10
    return User.query.filter(User.foo == 'bar').filter(db.or_(User.field_a.astext.in_(list_a), User.field_b.astext.in_(list_b))).filter(User.xyz.is_(None))