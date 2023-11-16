def survey_password_variables(survey_spec):
    if False:
        for i in range(10):
            print('nop')
    vars = []
    if 'spec' not in survey_spec:
        return vars
    for survey_element in survey_spec['spec']:
        if 'type' in survey_element and survey_element['type'] == 'password':
            vars.append(survey_element['variable'])
    return vars

def migrate_survey_passwords(apps, schema_editor):
    if False:
        print('Hello World!')
    'Take the output of the Job Template password list for all that\n    have a survey enabled, and then save it into the job model.\n    '
    Job = apps.get_model('main', 'Job')
    for job in Job.objects.iterator():
        if not job.job_template:
            continue
        jt = job.job_template
        if jt.survey_spec is not None and jt.survey_enabled:
            password_list = survey_password_variables(jt.survey_spec)
            hide_password_dict = {}
            for password in password_list:
                hide_password_dict[password] = '$encrypted$'
            job.survey_passwords = hide_password_dict
            job.save()