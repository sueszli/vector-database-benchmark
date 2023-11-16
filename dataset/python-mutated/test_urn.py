from pulumi import urn as urn_util

def test_parse_urn_with_name():
    if False:
        return 10
    res = urn_util._parse_urn('urn:pulumi:stack::project::pulumi:providers:aws::default_4_13_0')
    assert res.urn_name == 'default_4_13_0'
    assert res.typ == 'pulumi:providers:aws'
    assert res.pkg_name == 'pulumi'
    assert res.mod_name == 'providers'
    assert res.typ_name == 'aws'

def test_parse_urn_without_name():
    if False:
        print('Hello World!')
    res = urn_util._parse_urn('urn:pulumi:stack::project::pulumi:providers:aws')
    assert res.urn_name == ''
    assert res.typ == 'pulumi:providers:aws'
    assert res.pkg_name == 'pulumi'
    assert res.mod_name == 'providers'
    assert res.typ_name == 'aws'