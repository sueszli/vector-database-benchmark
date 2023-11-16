import json
import pytest
from itertools import count
from awx.main.utils.encryption import encrypt_value
from awx.main.models import Job, JobTemplate, JobLaunchConfig, WorkflowJobTemplate, Project, Inventory
ENCRYPTED_SECRET = encrypt_value('secret')

class DistinctParametrize(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._gen = count(0)

    def __call__(self, value):
        if False:
            for i in range(10):
                print('nop')
        return str(next(self._gen))

@pytest.mark.survey
class TestSurveyVariableValidation:

    def test_survey_answers_as_string(self, job_template_factory):
        if False:
            i = 10
            return i + 15
        objects = job_template_factory('job-template-with-survey', survey=[{'variable': 'var1', 'type': 'text'}], persisted=False)
        jt = objects.job_template
        user_extra_vars = json.dumps({'var1': 'asdf'})
        (accepted, ignored, errors) = jt._accept_or_ignore_job_kwargs(extra_vars=user_extra_vars)
        assert ignored.get('extra_vars', {}) == {}, [str(element) for element in errors]
        assert 'var1' in accepted['extra_vars']

    def test_job_template_survey_variable_validation(self, job_template_factory):
        if False:
            return 10
        objects = job_template_factory('survey_variable_validation', organization='org1', inventory='inventory1', credential='cred1', persisted=False)
        obj = objects.job_template
        obj.survey_spec = {'description': '', 'spec': [{'required': True, 'min': 0, 'default': '5', 'max': 1024, 'question_description': '', 'choices': '', 'variable': 'a', 'question_name': 'Whosyourdaddy', 'type': 'text'}], 'name': ''}
        obj.survey_enabled = True
        (accepted, rejected, errors) = obj.accept_or_ignore_variables({'a': 5})
        assert rejected == {'a': 5}
        assert accepted == {}
        assert str(errors['variables_needed_to_start'][0]) == "Value 5 for 'a' expected to be a string."

    def test_job_template_survey_default_variable_validation(self, job_template_factory):
        if False:
            while True:
                i = 10
        objects = job_template_factory('survey_variable_validation', organization='org1', inventory='inventory1', credential='cred1', persisted=False)
        obj = objects.job_template
        obj.survey_spec = {'description': '', 'spec': [{'required': True, 'min': 0, 'default': '2', 'max': 1024, 'question_description': '', 'choices': '', 'variable': 'a', 'question_name': 'float_number', 'type': 'float'}], 'name': ''}
        obj.survey_enabled = True
        (accepted, _, errors) = obj.accept_or_ignore_variables({'a': 2})
        assert accepted == {'a': 2.0}
        assert not errors

@pytest.fixture
def job(mocker):
    if False:
        for i in range(10):
            print('nop')
    ret = mocker.MagicMock(**{'decrypted_extra_vars.return_value': '{"secret_key": "my_password"}', 'display_extra_vars.return_value': '{"secret_key": "$encrypted$"}', 'extra_vars_dict': {'secret_key': 'my_password'}, 'pk': 1, 'job_template.pk': 1, 'job_template.name': '', 'created_by.pk': 1, 'created_by.username': 'admin', 'launch_type': 'manual', 'verbosity': 1, 'awx_meta_vars.return_value': {}, 'inventory.get_script_data.return_value': {}})
    ret.project = mocker.MagicMock(scm_revision='asdf1234')
    return ret

@pytest.fixture
def job_with_survey():
    if False:
        print('Hello World!')
    return Job(name='test-job-with-passwords', extra_vars=json.dumps({'submitter_email': 'foobar@redhat.com', 'secret_key': '6kQngg3h8lgiSTvIEb21', 'SSN': '123-45-6789'}), survey_passwords={'secret_key': '$encrypted$', 'SSN': '$encrypted$'})

@pytest.mark.survey
def test_job_survey_password_redaction(job_with_survey):
    if False:
        print('Hello World!')
    "Tests the Job model's funciton to redact passwords from\n    extra_vars - used when displaying job information"
    assert json.loads(job_with_survey.display_extra_vars()) == {'submitter_email': 'foobar@redhat.com', 'secret_key': '$encrypted$', 'SSN': '$encrypted$'}

@pytest.mark.survey
def test_survey_passwords_not_in_extra_vars():
    if False:
        return 10
    'Tests that survey passwords not included in extra_vars are\n    not included when displaying job information'
    job = Job(name='test-survey-not-in', extra_vars=json.dumps({'submitter_email': 'foobar@redhat.com'}), survey_passwords={'secret_key': '$encrypted$', 'SSN': '$encrypted$'})
    assert json.loads(job.display_extra_vars()) == {'submitter_email': 'foobar@redhat.com'}

def test_launch_config_has_unprompted_vars(survey_spec_factory):
    if False:
        return 10
    jt = JobTemplate(survey_enabled=True, survey_spec=survey_spec_factory(['question1', 'question2']))
    unprompted_config = JobLaunchConfig(extra_data={'question1': 'foobar', 'question4': 'foobar'})
    assert unprompted_config.has_unprompted(jt)
    allowed_config = JobLaunchConfig(extra_data={'question1': 'foobar', 'question2': 'foobar'})
    assert not allowed_config.has_unprompted(jt)

@pytest.mark.survey
def test_update_kwargs_survey_invalid_default(survey_spec_factory):
    if False:
        return 10
    spec = survey_spec_factory('var2')
    spec['spec'][0]['required'] = False
    spec['spec'][0]['min'] = 3
    spec['spec'][0]['default'] = 1
    jt = JobTemplate(name='test-jt', survey_spec=spec, survey_enabled=True, extra_vars='var2: 2')
    defaulted_extra_vars = jt._update_unified_job_kwargs({}, {})
    assert 'extra_vars' in defaulted_extra_vars
    assert json.loads(defaulted_extra_vars['extra_vars'])['var2'] == 2

@pytest.mark.survey
def test_display_survey_spec_encrypts_default(survey_spec_factory):
    if False:
        print('Hello World!')
    spec = survey_spec_factory('var2')
    spec['spec'][0]['type'] = 'password'
    spec['spec'][0]['default'] = 'some-default'
    jt = JobTemplate(name='test-jt', survey_spec=spec, survey_enabled=True)
    assert jt.display_survey_spec()['spec'][0]['default'] == '$encrypted$'

@pytest.mark.survey
@pytest.mark.parametrize('question_type,default,min,max,expect_use,expect_value', [('text', '', 0, 0, True, ''), ('text', '', 1, 0, False, 'N/A'), ('password', '', 1, 0, False, 'N/A'), ('multiplechoice', '', 0, 0, False, 'N/A'), ('multiplechoice', 'zeb', 0, 0, False, 'N/A'), ('multiplechoice', 'coffee', 0, 0, True, 'coffee'), ('multiselect', None, 0, 0, False, 'N/A'), ('multiselect', '', 0, 0, False, 'N/A'), ('multiselect', ['zeb'], 0, 0, False, 'N/A'), ('multiselect', ['milk'], 0, 0, True, ['milk']), ('multiselect', ['orange\nmilk'], 0, 0, False, 'N/A')])
def test_optional_survey_question_defaults(survey_spec_factory, question_type, default, min, max, expect_use, expect_value):
    if False:
        for i in range(10):
            print('nop')
    spec = survey_spec_factory([{'required': False, 'default': default, 'choices': 'orange\nmilk\nchocolate\ncoffee', 'variable': 'c', 'min': min, 'max': max, 'type': question_type}])
    jt = JobTemplate(name='test-jt', survey_spec=spec, survey_enabled=True)
    defaulted_extra_vars = jt._update_unified_job_kwargs({}, {})
    element = spec['spec'][0]
    if expect_use:
        assert jt._survey_element_validation(element, {element['variable']: element['default']}) == []
    else:
        assert jt._survey_element_validation(element, {element['variable']: element['default']})
    if expect_use:
        assert json.loads(defaulted_extra_vars['extra_vars'])['c'] == expect_value
    else:
        assert 'c' not in defaulted_extra_vars['extra_vars']

@pytest.mark.survey
@pytest.mark.parametrize('question_type,default,maxlen,kwargs,expected', [('text', None, 5, {}, {}), ('text', '', 5, {}, {'x': ''}), ('text', 'y', 5, {}, {'x': 'y'}), ('text', 'too-long', 5, {}, {}), ('password', None, 5, {}, {}), ('password', '', 5, {}, {'x': ''}), ('password', ENCRYPTED_SECRET, 5, {}, {}), ('password', ENCRYPTED_SECRET, 10, {}, {'x': ENCRYPTED_SECRET}), ('password', None, 5, {'extra_vars': {'x': '$encrypted$'}}, {}), ('password', '', 5, {'extra_vars': {'x': '$encrypted$'}}, {'x': ''}), ('password', None, 5, {'extra_vars': {'x': 'y'}}, {'x': 'y'}), ('password', '', 5, {'extra_vars': {'x': 'y'}}, {'x': 'y'}), ('password', 'foo', 5, {'extra_vars': {'x': 'y'}}, {'x': 'y'}), ('password', None, 5, {'extra_vars': {'x': ''}}, {'x': ''}), ('password', '', 5, {'extra_vars': {'x': ''}}, {'x': ''}), ('password', 'foo', 5, {'extra_vars': {'x': ''}}, {'x': ''}), ('password', ENCRYPTED_SECRET, 5, {'extra_vars': {'x': '$encrypted$'}}, {}), ('password', ENCRYPTED_SECRET, 10, {'extra_vars': {'x': '$encrypted$'}}, {'x': ENCRYPTED_SECRET})], ids=DistinctParametrize())
def test_survey_encryption_defaults(survey_spec_factory, question_type, default, maxlen, kwargs, expected):
    if False:
        print('Hello World!')
    spec = survey_spec_factory([{'required': True, 'variable': 'x', 'min': 0, 'max': maxlen, 'type': question_type}])
    if default is not None:
        spec['spec'][0]['default'] = default
    else:
        spec['spec'][0].pop('default', None)
    jt = JobTemplate(name='test-jt', survey_spec=spec, survey_enabled=True)
    extra_vars = json.loads(jt._update_unified_job_kwargs({}, kwargs).get('extra_vars'))
    assert extra_vars == expected

@pytest.mark.survey
@pytest.mark.django_db
class TestWorkflowSurveys:

    def test_update_kwargs_survey_defaults(self, survey_spec_factory):
        if False:
            return 10
        'Assure that the survey default over-rides a JT variable'
        spec = survey_spec_factory('var1')
        spec['spec'][0]['default'] = 3
        spec['spec'][0]['required'] = False
        wfjt = WorkflowJobTemplate.objects.create(name='test-wfjt', survey_spec=spec, survey_enabled=True, extra_vars='var1: 5')
        updated_extra_vars = wfjt._update_unified_job_kwargs({}, {})
        assert 'extra_vars' in updated_extra_vars
        assert json.loads(updated_extra_vars['extra_vars'])['var1'] == 3
        assert wfjt.can_start_without_user_input()

    def test_variables_needed_to_start(self, survey_spec_factory):
        if False:
            for i in range(10):
                print('nop')
        'Assure that variables_needed_to_start output contains mandatory vars'
        spec = survey_spec_factory(['question1', 'question2', 'question3'])
        spec['spec'][0]['required'] = False
        spec['spec'][1]['required'] = True
        spec['spec'][2]['required'] = False
        wfjt = WorkflowJobTemplate.objects.create(name='test-wfjt', survey_spec=spec, survey_enabled=True, extra_vars='question2: hiworld')
        assert wfjt.variables_needed_to_start == ['question2']
        assert not wfjt.can_start_without_user_input()

@pytest.mark.django_db
@pytest.mark.parametrize('provided_vars,valid', [({'tmpl_var': 'bar'}, True), ({'tmpl_var': 'bar2'}, False), ({'tmpl_var': 'bar', 'a': 2}, False), ({'tmpl_var': 'bar', False: 2}, False), ({'tmpl_var': 'bar', u'🐉': u'🐉'}, False)])
class TestExtraVarsNoPrompt:

    def process_vars_and_assert(self, tmpl, provided_vars, valid):
        if False:
            for i in range(10):
                print('nop')
        (prompted_fields, ignored_fields, errors) = tmpl._accept_or_ignore_job_kwargs(extra_vars=provided_vars)
        if valid:
            assert not ignored_fields
            assert not errors
        else:
            assert ignored_fields
            assert errors

    def test_jt_extra_vars_counting(self, provided_vars, valid):
        if False:
            for i in range(10):
                print('nop')
        jt = JobTemplate(name='foo', extra_vars={'tmpl_var': 'bar'}, project=Project(), project_id=42, playbook='helloworld.yml', inventory=Inventory(), inventory_id=42)
        (prompted_fields, ignored_fields, errors) = jt._accept_or_ignore_job_kwargs(extra_vars=provided_vars)
        self.process_vars_and_assert(jt, provided_vars, valid)

    def test_wfjt_extra_vars_counting(self, provided_vars, valid):
        if False:
            return 10
        wfjt = WorkflowJobTemplate.objects.create(name='foo', extra_vars={'tmpl_var': 'bar'})
        (prompted_fields, ignored_fields, errors) = wfjt._accept_or_ignore_job_kwargs(extra_vars=provided_vars)
        self.process_vars_and_assert(wfjt, provided_vars, valid)